// Entrypoint: speak agent-fun/bars-replay/v1 through the vendored shim and hand every completed
// bar to the RecoveryHunter strategy. This adapter is deliberately thin — all trading logic lives
// in strategy.ts, and all wire handling lives in vendor/makefun-bars.ts.
//
// Determinism is a hard requirement: the run is made twice and the two action logs must match
// byte for byte, so this file (and the strategy) never read the wall clock, the environment, the
// filesystem or Math.random. Every price, amount and depth comes from the batch in front of us,
// and time is the bar's close time. No state persists across a bar except what a batch carries.
//
// Order sizing uses the official SDK's integer amounts throughout. A buy is sized against the
// equity and cash the account reports on the bar it is decided, capped by the weight room and the
// minimum notional; a sell empties the whole position and is retried (exitPending) until the
// account confirms the position is gone, since a sell the market refuses (or a missing bar) must
// not silently strand a position. Buys allow 60 bps of slippage, sells 200 bps.

import { runBars, type OrderInput } from './vendor/makefun-bars.ts';
import { RecoveryHunter, EXIT, HOLD } from './strategy.ts';

const strategy = new RecoveryHunter();

// True once an exit has been requested but the account still reports a position, so the sell is
// sent again on the next bar until the replay confirms we are flat.
let exitPending = false;

await runBars({
  onBatch(state, batch): OrderInput[] {
    const symbol = state.hello.symbols[0]?.symbol;
    const bar = batch.bars.find((b) => b.symbol === symbol);
    // No bar for our symbol this minute: decide nothing (a gap is not a signal).
    if (!bar) return [];

    const held = state.positions.get(symbol);
    const positionValue = state.value(symbol, held?.qty ?? 0n);
    const quoteScale = 10 ** state.hello.capital.quote_scale;
    const priceScale = 10 ** state.symbols.get(symbol)!.price_scale;
    const equityUsd = Number(state.equity) / quoteScale;

    // Estimated round-trip cost of a full position: the fixed fee and slippage on both legs, plus
    // the price impact a trade worth the current equity would pay against the bar's close depth.
    const impact = bar.c_depth.reduce((sum, d) => sum + equityUsd / (Number(d) / 1e6), 0);
    const roundTrip = 2 * ((state.hello.fees.fee_bps + state.hello.fees.slippage_bps) / 10000 + impact);

    const decision = strategy.update({
      time: batch.t,
      price: Number(bar.c) / priceScale,
      equity: equityUsd,
      cash: Number(state.cash) / quoteScale,
      positionValue: Number(positionValue) / quoteScale,
      cost: Number(held?.cost ?? 0n) / quoteScale,
      roundTrip,
    });

    // The exit request lives until the account reports no position.
    if (!held?.qty) exitPending = false;
    if (decision === EXIT && held?.qty) exitPending = true;
    if (exitPending && held?.qty) {
      return [{ side: 'sell', symbol, qty: held.qty, max_slippage_bps: 200 }];
    }

    if (decision === HOLD) return [];

    // A positive decision is an entry weight: commit that share of equity, no more than the cash
    // on hand or the weight room, and at least the minimum notional.
    if (decision > 0 && !held?.qty) {
      const desired = BigInt(Math.floor(Number(state.equity) * decision));
      const amount = [desired, state.cash, state.weightRoom(symbol)].reduce((a, b) => (a < b ? a : b));
      if (amount >= BigInt(state.hello.fees.min_notional_quote_units)) {
        return [{ side: 'buy', symbol, notional_quote_units: amount, max_slippage_bps: 60 }];
      }
    }
    return [];
  },
});
