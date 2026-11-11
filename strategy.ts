// RecoveryHunter: a long-only dip-buy recovery trader for the agent-fun/bars-replay/v1
// protocol. The idea is simple and risk-first: when the market pulls back far enough and
// then visibly recovers, buy; when the recovery has run its course, sell. Everything runs on
// float64 (IEEE 754 double), so two runs over the same bars give byte-identical orders.
//
// The decision surface has two halves. While flat we look for an entry: either a deep dip from
// the lookback high, or a shallower dip from the last half hour's high while price also trades
// below its slow average. Both must be followed by a rebound that clears a floor sized by recent
// volatility and the estimated round-trip cost (fees + slippage + price impact). While holding
// we manage the exit: a hard stop, an orderly-trend extension behind a trailing stop, a
// take-profit, and a patience timeout. A 12% drawdown from peak equity permanently halts new
// entries (exits are always allowed).

export interface Params {
  /** EMA span (minutes) of the fast average. */
  fastSpan: number;
  /** EMA span (minutes) of the slow average. */
  slowSpan: number;
  /** How many minutes of price history to keep (also the horizon of the deep-dip high). */
  lookbackMinutes: number;
  /** Minimum bars of history before the first entry may be taken. */
  warmupBars: number;
  /** Minimum deep dip (fraction below the lookback high) to qualify. */
  dip: number;
  /** Minimum shallow dip (fraction below the 30-minute high) to qualify. */
  secondaryDip: number;
  /** Minimum rebound floor (fraction) before an entry confirms. */
  rebound: number;
  /** Take-profit level (fraction over cost) in normal mode. */
  take: number;
  /** Trailing stop (fraction below the high since entry) in normal mode. */
  trail: number;
  /** Hard stop loss (fraction below cost). */
  stop: number;
  /** Minutes after entry before a non-orderly, unprofitable position is timed out. */
  patience: number;
  /** Minutes to wait after an exit before a new entry is allowed. */
  cooldown: number;
  /** Fraction of equity to commit on an entry (<= 1). */
  allocation: number;
  /** Minimum fast/slow gap (fraction) that counts as an orderly up-trend. */
  trendGap: number;
  /** Minimum net-move / path-length over the last hour to count as orderly. */
  efficiency: number;
  /** Round-trip-cost multiplier for the rebound floor. */
  confirmCost: number;
  /** Volatility multiplier for the rebound floor. */
  confirmVol: number;
  /** Minutes back for the rebound's local low. */
  recoveryWindow: number;
}

export const DEFAULT_PARAMS: Params = {
  fastSpan: 18,
  slowSpan: 120,
  lookbackMinutes: 120,
  warmupBars: 18,
  dip: 0.009,
  secondaryDip: 0.0045,
  rebound: 0.0005,
  take: 0.007,
  trail: 0.004,
  stop: 0.018,
  patience: 480,
  cooldown: 10,
  allocation: 0.95,
  trendGap: 0.0015,
  efficiency: 0.45,
  confirmCost: 0.5,
  confirmVol: 1.5,
  recoveryWindow: 12,
};

/** One per-bar observation, already converted to USDC (and fractions) by the adapter. */
export interface Observation {
  /** Bar close time, milliseconds. */
  time: number;
  /** Bar close price, USDC. */
  price: number;
  /** Current equity (cash + position marked at close), USDC. */
  equity: number;
  /** Cash, USDC. */
  cash: number;
  /** Position value at the close, USDC. */
  positionValue: number;
  /** The account's reported cost basis for the position, USDC (0 when flat). */
  cost: number;
  /** Estimated round-trip cost (fees + slippage + impact, both legs), as a fraction. */
  roundTrip: number;
}

/** Return convention: null = hold / no action, 0 = exit, a positive number = entry weight. */
export type Decision = number | null;

export const HOLD: null = null;
export const EXIT = 0;

interface Point {
  t: number;
  p: number;
}

export class RecoveryHunter {
  readonly params: Params;

  // Indicators and rolling state.
  private fast = 0; // exponential moving average (fast span), seeded on the first close
  private slow = 0; // exponential moving average (slow span), seeded on the first close
  private vol = 0.0005; // volatility estimate, |log return| / sqrt(elapsed) blended
  private prior = 0; // previous close (0 before the first bar)
  private time = 0; // previous observation time (0 before the first bar)
  private history: Point[] = []; // {t, p} trimmed to the lookback window

  // Position lifecycle.
  private peakEquity = 1000; // running peak of marked equity, USDC
  private halted = false; // set once equity < 0.88 * peak; blocks new entries forever
  private wasHolding = false; // whether the previous observation held a position
  private entryTime = 0; // time the current position first appeared (ms)
  private highSinceEntry = 0; // highest close since entry
  private lastExit = -1e15; // time of the last exit (ms); far in the past initially
  private mode: '' | 'trend' | 'recovery' = ''; // current hold mode

  constructor(params: Partial<Params> = {}) {
    this.params = { ...DEFAULT_PARAMS, ...params };
  }

  update(x: Observation): Decision {
    const c = this.params;
    const p = x.price;

    // Minutes since the previous observation (at least one, and one before the first bar).
    const elapsed = this.time ? Math.max(1, (x.time - this.time) / 60000) : 1;
    // Absolute log return since the previous close.
    const ret = this.prior ? Math.abs(Math.log(p / this.prior)) : 0;

    // Capture the pre-update values the trend check needs, then advance the averages.
    const oldFast = this.fast || p;
    const previous = this.prior || p;
    this.fast = this.fast ? this.fast + (1 - Math.exp(-elapsed / c.fastSpan)) * (p - this.fast) : p;
    this.slow = this.slow ? this.slow + (1 - Math.exp(-elapsed / c.slowSpan)) * (p - this.slow) : p;
    this.vol = 0.95 * this.vol + 0.05 * ret / Math.sqrt(elapsed);
    this.prior = p;
    this.time = x.time;

    // Rolling history of closes, trimmed to the lookback window.
    this.history.push({ t: x.time, p });
    while (this.history.length && this.history[0].t < x.time - c.lookbackMinutes * 60000) {
      this.history.shift();
    }

    // Peak-equity drawdown guard.
    this.peakEquity = Math.max(this.peakEquity, x.equity);
    if (x.equity < this.peakEquity * 0.88) this.halted = true;

    // Recognise entry/exit only from the account's reported position, never from a requested
    // order: the position value is derived from the reported qty, so a fill the market refused
    // (or one still in flight) never looks like a position here. The >1 USDC threshold is the
    // same "does a position exist" test, since any filled buy is worth at least the minimum
    // notional of 1 USDC.
    const held = x.positionValue > 1;
    if (held && !this.wasHolding) {
      this.entryTime = x.time;
      this.highSinceEntry = p;
    }
    if (!held && this.wasHolding) {
      this.lastExit = x.time;
      this.mode = '';
    }
    this.wasHolding = held;

    if (this.halted) return held ? EXIT : HOLD;
    if (this.history.length < c.warmupBars) return HOLD;

    // Shared per-bar quantities.
    const recent = this.history.filter((h) => h.t >= x.time - c.recoveryWindow * 60000);
    const low = Math.min(...recent.map((h) => h.p));
    const top = Math.max(...this.history.map((h) => h.p));
    const trend = this.fast / this.slow - 1;
    const hourRows = this.history.filter((h) => h.t >= x.time - 60 * 60000);
    let path = 0;
    for (let i = 1; i < hourRows.length; i++) path += Math.abs(hourRows[i].p - hourRows[i - 1].p);
    const efficiency = path ? Math.abs(p - hourRows[0].p) / path : 0;
    // "Orderly" up-trend: fast clearly above slow, price above fast, fast still rising, and the
    // last hour's move is directional rather than choppy.
    const orderly = trend > c.trendGap && p > this.fast && this.fast > oldFast && efficiency > c.efficiency;

    if (held) {
      this.highSinceEntry = Math.max(this.highSinceEntry, p);
      const profit = x.cost > 0 ? x.positionValue / x.cost - 1 : 0;
      const age = (x.time - this.entryTime) / 60000;
      const trailing = Math.max(c.trail, 4 * this.vol, 1.5 * x.roundTrip);

      if (profit < -c.stop) return EXIT;

      // A sufficiently orderly trend may extend a profitable recovery.
      if (profit > Math.max(c.take, x.roundTrip) && orderly) this.mode = 'trend';

      if (this.mode === 'trend') {
        if (1 - p / this.highSinceEntry > trailing && (p < this.fast || trend < 0)) return EXIT;
      } else {
        if (profit >= Math.max(c.take, x.roundTrip * 0.75)) return EXIT;
        if (profit > x.roundTrip && 1 - p / this.highSinceEntry > trailing) return EXIT;
      }
      if (age > c.patience && profit < x.roundTrip && !orderly) return EXIT;
      return HOLD;
    }

    if (x.time - this.lastExit < c.cooldown * 60000) return HOLD;

    const movement = Math.max(c.rebound, c.confirmVol * this.vol, c.confirmCost * x.roundTrip);
    const dipThreshold = Math.max(c.dip, x.roundTrip * 1.2, 3 * this.vol);
    const deepDip = 1 - p / top > dipThreshold;
    const shortTop = Math.max(...this.history.filter((h) => h.t >= x.time - 30 * 60000).map((h) => h.p));
    // The smaller pullback must also be below the slow average.
    const shortDip = p < this.slow && 1 - p / shortTop > Math.max(c.secondaryDip, x.roundTrip * 1.2, 3 * this.vol);

    // A dip only becomes an entry once the rebound clears the floor and price is above the prior close.
    if ((deepDip || shortDip) && p / low - 1 > movement && p > previous) {
      this.mode = 'recovery';
      return c.allocation;
    }
    return HOLD;
  }
}
