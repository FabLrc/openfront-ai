import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const openfront = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../openfront",
);
const pkg = (name: string) => path.join(openfront, "packages", name, "src");

export default defineConfig({
  resolve: {
    alias: [
      { find: /^@openfront\/zbin$/, replacement: pkg("zbin") + "/index.ts" },
      { find: /^@openfront\/([^/]+)\/(.*)$/, replacement: pkg("$1") + "/$2" },
      { find: /^resources\/(.*)$/, replacement: openfront + "/resources/$1" },
    ],
  },
  test: {
    testTimeout: 300_000,
  },
});
