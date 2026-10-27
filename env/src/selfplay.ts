/**
 * Joue une partie entière sans agent (uniquement des Nations) et affiche
 * le vainqueur et le hash final. Utile pour mesurer la vitesse du moteur.
 *
 *   npx tsx src/selfplay.ts [--map world] [--size Compact] [--nations 8] [--seed s]
 */
import { parseArgs } from "node:util";
import { OpenFrontGame } from "./game";

const { values } = parseArgs({
  options: {
    map: { type: "string", default: "world" },
    size: { type: "string", default: "Compact" },
    nations: { type: "string", default: "8" },
    difficulty: { type: "string", default: "Medium" },
    seed: { type: "string", default: "openfront-ai" },
  },
});

const log = console.log;
console.log = console.debug = () => {}; // le moteur est bavard
const t0 = performance.now();
const game = await OpenFrontGame.create({
  map: values.map,
  size: values.size as "Compact" | "Normal",
  nations: parseInt(values.nations, 10),
  difficulty: values.difficulty as "Medium",
  seed: values.seed,
});
let s = game.state();
while (!s.done) {
  s = game.step(1000);
  const top = [...s.players].sort((a, b) => b.tiles - a.tiles)[0];
  log(`tick ${s.tick}  vivants ${s.alivePlayers}  1er ${top.name} (${top.tiles} tuiles)`);
}
const ms = performance.now() - t0;
log(`Vainqueur : ${s.winner} au tick ${s.tick}, hash ${s.hash}, ${(ms / 1000).toFixed(1)} s (${Math.round(s.tick / (ms / 1000))} ticks/s)`);
