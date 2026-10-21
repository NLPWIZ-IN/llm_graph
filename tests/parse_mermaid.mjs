import { JSDOM } from "jsdom";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const dom = new JSDOM("");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
const { default: mermaid } = await import("mermaid");
mermaid.initialize({
  startOnLoad: false,
  securityLevel: "strict",
  maxEdges: 10000,
  maxTextSize: 1000000,
});
for (const name of await readdir(process.argv[2])) {
  await mermaid.parse(await readFile(join(process.argv[2], name), "utf8"));
  console.log(`PASS ${name}`);
}
