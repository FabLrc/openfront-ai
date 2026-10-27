import { beforeAll, describe, expect, it } from "vitest";
import { GameState, OpenFrontGame, ResetConfig } from "../src/game";

beforeAll(() => {
  // Le moteur journalise beaucoup.
  console.log = console.debug = console.info = () => {};
});

async function playToEnd(config: ResetConfig): Promise<GameState> {
  const game = await OpenFrontGame.create(config);
  let state = game.state();
  while (!state.done) state = game.step(500);
  return state;
}

describe("déterminisme", () => {
  const config: ResetConfig = { map: "world", size: "Compact", nations: 8, seed: "test-determinisme" };

  it("deux parties avec la même graine donnent le même vainqueur et le même hash", async () => {
    const a = await playToEnd(config);
    const b = await playToEnd(config);
    expect(a.winner).not.toBeNull();
    expect(a.hash).not.toBeNull();
    expect(b.winner).toBe(a.winner);
    expect(b.tick).toBe(a.tick);
    expect(b.hash).toBe(a.hash);
  });

  it("step s'arrête au tick de la victoire", async () => {
    const game = await OpenFrontGame.create(config);
    let state = game.state();
    while (!state.done) state = game.step(1000);
    expect(game.step(10).tick).toBe(state.tick);
  });
});
