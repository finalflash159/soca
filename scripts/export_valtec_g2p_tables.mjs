import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";
import vm from "node:vm";

const repoRoot = resolve(import.meta.dirname, "..");
const sourcePath = resolve(
  process.argv[2] ?? `${repoRoot}/external/valtec-tts/deployments/web/vietnamese_g2p.js`,
);
const outputPath = resolve(
  process.argv[3] ?? `${repoRoot}/soca/tts/valtec/g2p_tables.json`,
);
const sourceRelative = relative(repoRoot, sourcePath);
if (sourceRelative === ".." || sourceRelative.startsWith(`..${sep}`)) {
  throw new Error(`G2P source must be inside the repository: ${sourcePath}`);
}
const source = await readFile(sourcePath, "utf8");
const expose = `
globalThis.__socaTables = {
  onsets: Cus_onsets, nuclei: Cus_nuclei, offglides: Cus_offglides,
  onglides: Cus_onglides, onoffglides: Cus_onoffglides, codas: Cus_codas,
  tones: Cus_tones_p, gi: Cus_gi, qu: Cus_qu
};`;
const context = Object.create(null);
vm.createContext(context, {
  codeGeneration: { strings: false, wasm: false },
});
vm.runInContext(source + expose, context, {
  filename: sourcePath,
  timeout: 1000,
});

const expectedTables = [
  "onsets", "nuclei", "offglides", "onglides", "onoffglides",
  "codas", "tones", "gi", "qu",
];
for (const name of expectedTables) {
  const table = context.__socaTables?.[name];
  if (!table || typeof table !== "object" || Array.isArray(table) || !Object.keys(table).length) {
    throw new Error(`Missing or empty Valtec G2P table: ${name}`);
  }
}

function sortObject(value) {
  if (Array.isArray(value)) return value.map(sortObject);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, sortObject(value[key])]),
    );
  }
  return value;
}

const payload = {
  schema_version: 1,
  generated_from: sourceRelative.split(sep).join("/"),
  source_sha256: createHash("sha256").update(source).digest("hex"),
  tables: sortObject(context.__socaTables),
};
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
