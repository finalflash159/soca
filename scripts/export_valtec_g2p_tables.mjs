import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import vm from "node:vm";

const repoRoot = resolve(import.meta.dirname, "..");
const sourceRepository = "https://github.com/tronghieuit/valtec-tts";
const sourceRevision = "a5e77a138960c1101c022a61614d6ee72aeccadc";
const sourceFile = "deployments/web/vietnamese_g2p.js";
const sourceUrl = `https://raw.githubusercontent.com/tronghieuit/valtec-tts/${sourceRevision}/${sourceFile}`;
const outputPath = resolve(
  process.argv[2] ?? `${repoRoot}/soca/tts/valtec/g2p_tables.json`,
);
const response = await fetch(sourceUrl);
if (!response.ok) {
  throw new Error(
    `Failed to download Valtec G2P source (${response.status}): ${sourceUrl}`,
  );
}
const source = await response.text();
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
  filename: sourceUrl,
  timeout: 1000,
});

const expectedTables = [
  "onsets",
  "nuclei",
  "offglides",
  "onglides",
  "onoffglides",
  "codas",
  "tones",
  "gi",
  "qu",
];
for (const name of expectedTables) {
  const table = context.__socaTables?.[name];
  if (
    !table ||
    typeof table !== "object" ||
    Array.isArray(table) ||
    !Object.keys(table).length
  ) {
    throw new Error(`Missing or empty Valtec G2P table: ${name}`);
  }
}

function sortObject(value) {
  if (Array.isArray(value)) return value.map(sortObject);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, sortObject(value[key])]),
    );
  }
  return value;
}

const payload = {
  schema_version: 1,
  generated_from: sourceUrl,
  source_repository: sourceRepository,
  source_revision: sourceRevision,
  source_path: sourceFile,
  source_sha256: createHash("sha256").update(source).digest("hex"),
  tables: sortObject(context.__socaTables),
};
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
