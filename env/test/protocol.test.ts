import { describe, expect, it } from "vitest";
import { encodeFrame, FrameReader } from "../src/protocol";

describe("trames préfixées par leur longueur", () => {
  it("reconstruit des messages coupés ou groupés", () => {
    const stream = Buffer.concat([encodeFrame({ cmd: "reset" }), encodeFrame({ cmd: "step", ticks: 10 })]);
    const reader = new FrameReader();
    const got = [...reader.push(stream.subarray(0, 7)), ...reader.push(stream.subarray(7))];
    expect(got).toEqual([{ cmd: "reset" }, { cmd: "step", ticks: 10 }]);
  });
});
