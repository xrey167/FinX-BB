import { buildApp } from "./app.js";
import { resolveListenHost } from "./listen.js";

const port = Number(process.env.PORT ?? 3001);
const app = buildApp();

try {
  const host = resolveListenHost(process.env);
  await app.listen({ port, host });
} catch (error) {
  app.log.error(error);
  process.exitCode = 1;
}
