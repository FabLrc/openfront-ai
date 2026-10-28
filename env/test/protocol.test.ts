import { describe, expect, it } from "vitest";
import { encodeFrame, FrameReader } from "../src/protocol";

describe("trames préfixées par leur longueur", () => {
  it("reconstruit des messages coupés ou groupés", () => {
    const stream = Buffer.concat([encodeFrame({ cmd: "reset" }), encodeFrame({ cmd: "step", ticks: 10 })]);
    const reader = new FrameReader();
    const got = [...reader.push(stream.subarray(0, 7)), ...reader.push(stream.subarray(7))];
    expect(got.map((f) => f.header)).toEqual([{ cmd: "reset" }, { cmd: "step", ticks: 10 }]);
  });

  it("aller-retour des tableaux binaires", () => {
    const spatial = Uint8Array.from({ length: 2 * 3 * 5 }, (_, i) => (i * 37) % 256);
    const players = Float32Array.from([1.5, -2.25, Math.PI, 0, 1e6]);
    const ids = Int32Array.from([-7, 0, 123456]);
    const frame = encodeFrame(
      { ok: true, note: "é" }, // en-tête de longueur impaire : teste l'alignement
      {
        "agent_0/spatial": { data: spatial, shape: [2, 3, 5] },
        "agent_0/players": { data: players, shape: [5] },
        ids: { data: ids, shape: [3] },
      },
    );
    const [decoded] = new FrameReader().push(frame);
    expect(decoded.header).toEqual({ ok: true, note: "é" });
    expect(decoded.buffers["agent_0/spatial"].shape).toEqual([2, 3, 5]);
    expect([...decoded.buffers["agent_0/spatial"].data]).toEqual([...spatial]);
    expect([...decoded.buffers["agent_0/players"].data]).toEqual([...players]);
    expect(decoded.buffers["ids"].data).toBeInstanceOf(Int32Array);
    expect([...decoded.buffers["ids"].data]).toEqual([...ids]);
  });
});
