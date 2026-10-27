/**
 * Trames du socket d'environnement : un entier non signé 32 bits big-endian
 * (longueur en octets) suivi d'un message JSON UTF-8.
 */
import { Socket } from "node:net";

export const MAX_FRAME_BYTES = 64 * 1024 * 1024;

export function encodeFrame(message: unknown): Buffer {
  const body = Buffer.from(JSON.stringify(message), "utf8");
  const header = Buffer.alloc(4);
  header.writeUInt32BE(body.length, 0);
  return Buffer.concat([header, body]);
}

/** Découpe le flux d'un socket en messages JSON, dans l'ordre d'arrivée. */
export class FrameReader {
  private buffer = Buffer.alloc(0);

  push(chunk: Buffer): unknown[] {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    const messages: unknown[] = [];
    while (this.buffer.length >= 4) {
      const length = this.buffer.readUInt32BE(0);
      if (length > MAX_FRAME_BYTES) {
        throw new Error(`trame trop grande : ${length} octets`);
      }
      if (this.buffer.length < 4 + length) break;
      messages.push(JSON.parse(this.buffer.subarray(4, 4 + length).toString("utf8")));
      this.buffer = this.buffer.subarray(4 + length);
    }
    return messages;
  }
}

export function sendFrame(socket: Socket, message: unknown): void {
  socket.write(encodeFrame(message));
}
