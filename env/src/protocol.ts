/**
 * Trames du socket d'environnement.
 *
 *   u32 BE  longueur totale du reste de la trame
 *   u32 BE  longueur de l'en-tête JSON
 *   ...     en-tête JSON UTF-8
 *   ...     tableaux binaires (little-endian), chacun aligné sur 8 octets
 *
 * L'en-tête décrit les tableaux dans `__buffers` : nom, type (uint8, int32,
 * float32…), forme, position et taille en octets. Les gros tableaux
 * (observations, masques) ne passent donc jamais par JSON. Les requêtes du
 * client n'ont en général que l'en-tête.
 */
import { Socket } from "node:net";

export const MAX_FRAME_BYTES = 256 * 1024 * 1024;

export type TypedArray = Uint8Array | Uint16Array | Int32Array | Float32Array;
export type Buffers = Record<string, { data: TypedArray; shape: number[] }>;

const DTYPES: Record<string, new (b: ArrayBuffer, o: number, n: number) => TypedArray> = {
  uint8: Uint8Array,
  uint16: Uint16Array,
  int32: Int32Array,
  float32: Float32Array,
};

function dtypeOf(a: TypedArray): string {
  if (a instanceof Uint8Array) return "uint8";
  if (a instanceof Uint16Array) return "uint16";
  if (a instanceof Int32Array) return "int32";
  return "float32";
}

interface BufferSpec {
  name: string;
  dtype: string;
  shape: number[];
  offset: number;
  nbytes: number;
}

const pad8 = (n: number) => (n + 7) & ~7;

export function encodeFrame(header: object, buffers: Buffers = {}): Buffer {
  const specs: BufferSpec[] = [];
  let offset = 0;
  for (const [name, { data, shape }] of Object.entries(buffers)) {
    specs.push({ name, dtype: dtypeOf(data), shape, offset, nbytes: data.byteLength });
    offset = pad8(offset + data.byteLength);
  }
  const json = Buffer.from(
    JSON.stringify(specs.length > 0 ? { ...header, __buffers: specs } : header),
    "utf8",
  );
  const blobStart = pad8(8 + json.length);
  const out = Buffer.alloc(blobStart + offset);
  out.writeUInt32BE(out.length - 4, 0);
  out.writeUInt32BE(json.length, 4);
  json.copy(out, 8);
  for (const spec of specs) {
    const { data } = buffers[spec.name];
    Buffer.from(data.buffer, data.byteOffset, data.byteLength).copy(out, blobStart + spec.offset);
  }
  return out;
}

export interface Frame {
  header: Record<string, unknown>;
  buffers: Buffers;
}

/** Décode le corps d'une trame (sans les 4 premiers octets de longueur). */
export function decodeBody(body: Buffer): Frame {
  const jsonLength = body.readUInt32BE(0);
  const header = JSON.parse(body.subarray(4, 4 + jsonLength).toString("utf8"));
  const buffers: Buffers = {};
  const specs: BufferSpec[] = header.__buffers ?? [];
  delete header.__buffers;
  // Même alignement que encodeFrame, en comptant les 4 octets de longueur.
  const blobStart = pad8(8 + jsonLength) - 4;
  for (const spec of specs) {
    const Ctor = DTYPES[spec.dtype];
    if (Ctor === undefined) throw new Error(`type de tableau inconnu : ${spec.dtype}`);
    // Copie : la trame d'origine n'est pas forcément alignée en mémoire.
    const bytes = new Uint8Array(spec.nbytes);
    body.copy(bytes, 0, blobStart + spec.offset, blobStart + spec.offset + spec.nbytes);
    const n = spec.nbytes / (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    buffers[spec.name] = { data: new Ctor(bytes.buffer, 0, n), shape: spec.shape };
  }
  return { header, buffers };
}

/** Découpe le flux d'un socket en trames, dans l'ordre d'arrivée. */
export class FrameReader {
  private buffer: Buffer = Buffer.alloc(0);

  push(chunk: Buffer): Frame[] {
    this.buffer = this.buffer.length === 0 ? chunk : Buffer.concat([this.buffer, chunk]);
    const frames: Frame[] = [];
    while (this.buffer.length >= 4) {
      const length = this.buffer.readUInt32BE(0);
      if (length > MAX_FRAME_BYTES) {
        throw new Error(`trame trop grande : ${length} octets`);
      }
      if (this.buffer.length < 4 + length) break;
      frames.push(decodeBody(this.buffer.subarray(4, 4 + length)));
      this.buffer = this.buffer.subarray(4 + length);
    }
    return frames;
  }
}

export function sendFrame(socket: Socket, header: object, buffers?: Buffers): void {
  socket.write(encodeFrame(header, buffers));
}
