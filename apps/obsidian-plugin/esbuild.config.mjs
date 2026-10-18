import esbuild from "esbuild";

const watch = process.argv.includes("--watch");
const context = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  external: ["obsidian", "electron", "@codemirror/state", "@codemirror/view"],
  platform: "browser",
  format: "cjs",
  target: "es2020",
  outfile: "main.js",
  logLevel: "info",
});
if (watch) {
  await context.watch();
} else {
  await context.rebuild();
  await context.dispose();
}
