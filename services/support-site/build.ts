import worker from "./src/index.ts";

await Bun.write(`${import.meta.dir}/public/index.html`, await worker.fetch(new Request("https://betabuddy.dozycat.ai/")).text());
await Bun.write(`${import.meta.dir}/public/privacy/index.html`, await worker.fetch(new Request("https://betabuddy.dozycat.ai/privacy")).text());
