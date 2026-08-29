import { complete } from "./protocol.mjs";
import { loadPackage } from "./package.mjs";
import { sdkVersions } from "./version.mjs";

export class Engine {
  constructor(pkg) {
    this.package = pkg;
    this.closed = false;
  }

  static load(packageDir, options) {
    return new Engine(loadPackage(packageDir, options));
  }

  capabilities() {
    return this.package.capabilities(sdkVersions());
  }

  createSession() {
    if (this.closed) throw new Error("engine is closed");
    const caps = this.capabilities();
    return {
      complete(request) {
        return complete(request, caps);
      },
      run(request, maxTurns = 1) {
        const turns = [];
        let stopped = "max_turns";
        for (let i = 0; i < Math.max(1, maxTurns); i += 1) {
          const turn = complete(request, caps);
          turns.push(turn);
          if (turn.error) {
            stopped = turn.error.id === "cancelled" ? "cancelled" : "error";
            break;
          }
          if (turn.refuse) {
            stopped = "refuse";
            break;
          }
          if (turn.function_calls?.length) {
            stopped = "call";
            break;
          }
        }
        return {
          wire_version: sdkVersions().wire_version,
          ok: turns.length > 0 && turns.every((t) => t.ok),
          turns,
          stopped_reason: stopped,
        };
      },
    };
  }
}
