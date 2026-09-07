import { describe, expect, it } from "vitest";

import { resolveListenHost } from "../src/listen.js";

describe("listen address", () => {
  it("defaults to IPv4 loopback", () => {
    expect(resolveListenHost({})).toBe("127.0.0.1");
  });

  it.each(["127.0.0.1", "::1", "localhost"])(
    "allows loopback host %s",
    (host) => {
      expect(resolveListenHost({ HOST: host })).toBe(host);
    },
  );

  it("requires an explicit opt-in for a remote bind", () => {
    expect(() => resolveListenHost({ HOST: "0.0.0.0" })).toThrow(
      /INSECURE_REMOTE_DEMO=1/,
    );
    expect(
      resolveListenHost({ HOST: "0.0.0.0", INSECURE_REMOTE_DEMO: "1" }),
    ).toBe("0.0.0.0");
  });
});
