const loopbackHosts = new Set(["127.0.0.1", "::1", "localhost"]);

type ListenEnvironment = {
  HOST?: string | undefined;
  INSECURE_REMOTE_DEMO?: string | undefined;
};

export function resolveListenHost(environment: ListenEnvironment): string {
  const host = environment.HOST ?? "127.0.0.1";
  if (
    !loopbackHosts.has(host.toLowerCase()) &&
    environment.INSECURE_REMOTE_DEMO !== "1"
  ) {
    throw new Error(
      "Remote bind requires the explicit demo opt-in INSECURE_REMOTE_DEMO=1",
    );
  }
  return host;
}
