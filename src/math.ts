export type Dimension = 2 | 3;
export type Vector = number[];
export type Matrix = number[][];
export type PivotStrategy = "residual" | "greedy" | "uniform";

export type Example = {
  id: string;
  label: string;
  dimension: Dimension;
  features: Matrix;
};

export type PivotState = {
  approximation: Matrix;
  residual: Matrix;
  pivots: number[];
};

const TOLERANCE = 1e-9;

export const EXAMPLES: Record<Dimension, Example> = {
  2: {
    id: "features-2d",
    label: "2D feature vectors",
    dimension: 2,
    features: [
      [2.3, 0.5],
      [1.15, 1.9],
      [-1.35, 1.55],
      [-2.05, -0.65],
      [0.55, -1.85],
    ],
  },
  3: {
    id: "features-3d",
    label: "3D feature vectors",
    dimension: 3,
    features: [
      [2.15, 0.35, 0.25],
      [1.1, 1.75, 0.55],
      [-1.35, 1.35, 0.85],
      [-1.7, -0.55, 1.45],
      [0.45, -1.8, 0.9],
      [0.8, 0.35, -1.7],
    ],
  },
};

export function gramMatrix(features: Matrix): Matrix {
  return features.map((left) => features.map((right) => dot(left, right)));
}

export function createInitialState(features: Matrix): PivotState {
  const matrixSize = features.length;

  return {
    approximation: zeroMatrix(matrixSize),
    residual: gramMatrix(features),
    pivots: [],
  };
}

export function advanceState(state: PivotState, pivot: number): PivotState | null {
  const denominator = state.residual[pivot][pivot];
  if (denominator <= TOLERANCE) {
    return null;
  }

  const column = state.residual.map((row) => row[pivot]);
  const update = column.map((left) =>
    column.map((right) => (left * right) / denominator),
  );

  return {
    approximation: combineMatrices(state.approximation, update, 1),
    residual: cleanResidual(combineMatrices(state.residual, update, -1)),
    pivots: [...state.pivots, pivot],
  };
}

export function residualDiagonal(residual: Matrix): Vector {
  return residual.map((row, index) => Math.max(0, row[index]));
}

export function trace(matrix: Matrix): number {
  return matrix.reduce((sum, row, index) => sum + row[index], 0);
}

export function residualProbabilities(residual: Matrix): Vector {
  const diagonal = residualDiagonal(residual);
  const total = diagonal.reduce((sum, value) => sum + value, 0);

  if (total <= TOLERANCE) {
    return diagonal.map(() => 0);
  }

  return diagonal.map((value) => value / total);
}

export function eligiblePivots(state: PivotState): number[] {
  const selected = new Set(state.pivots);
  return residualDiagonal(state.residual)
    .map((value, index) => ({ value, index }))
    .filter(({ value, index }) => value > TOLERANCE && !selected.has(index))
    .map(({ index }) => index);
}

export function choosePivot(
  state: PivotState,
  strategy: PivotStrategy,
  seedText: string,
  manualPivot: number | null,
): number | null {
  const eligible = eligiblePivots(state);
  if (eligible.length === 0) {
    return null;
  }

  if (manualPivot !== null && eligible.includes(manualPivot)) {
    return manualPivot;
  }

  if (strategy === "greedy") {
    return eligible.reduce((best, index) =>
      state.residual[index][index] > state.residual[best][best] ? index : best,
    );
  }

  const random = deterministicRandom(
    `${seedText}:${strategy}:${state.pivots.join("-") || "start"}`,
  );

  if (strategy === "uniform") {
    return eligible[Math.floor(random * eligible.length)];
  }

  const probabilities = residualProbabilities(state.residual);
  let cumulative = 0;

  for (const index of eligible) {
    cumulative += probabilities[index];
    if (random <= cumulative) {
      return index;
    }
  }

  return eligible[eligible.length - 1];
}

export function orthonormalBasis(features: Matrix, pivots: number[]): Matrix {
  const basis: Matrix = [];

  for (const pivot of pivots) {
    let candidate = [...features[pivot]];

    for (const vector of basis) {
      candidate = subtract(candidate, scale(vector, dot(candidate, vector)));
    }

    const length = norm(candidate);
    if (length > TOLERANCE) {
      basis.push(scale(candidate, 1 / length));
    }
  }

  return basis;
}

export function projectFeatures(features: Matrix, basis: Matrix): Matrix {
  return features.map((vector) => projectVector(vector, basis));
}

export function projectVector(vector: Vector, basis: Matrix): Vector {
  return basis.reduce(
    (projection, basisVector) =>
      add(projection, scale(basisVector, dot(vector, basisVector))),
    vector.map(() => 0),
  );
}

export function subtractMatrices(left: Matrix, right: Matrix): Matrix {
  return left.map((row, rowIndex) =>
    row.map((value, columnIndex) => value - right[rowIndex][columnIndex]),
  );
}

export function dot(left: Vector, right: Vector): number {
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}

export function add(left: Vector, right: Vector): Vector {
  return left.map((value, index) => value + right[index]);
}

export function subtract(left: Vector, right: Vector): Vector {
  return left.map((value, index) => value - right[index]);
}

export function scale(vector: Vector, factor: number): Vector {
  return vector.map((value) => value * factor);
}

export function normSquared(vector: Vector): number {
  return dot(vector, vector);
}

export function norm(vector: Vector): number {
  return Math.sqrt(normSquared(vector));
}

export function formatNumber(value: number): string {
  const normalized = Math.abs(value) < 1e-10 ? 0 : value;
  return normalized.toLocaleString("en-US", {
    maximumFractionDigits: 4,
  });
}

function zeroMatrix(size: number): Matrix {
  return Array.from({ length: size }, () => Array(size).fill(0));
}

function combineMatrices(left: Matrix, right: Matrix, sign: 1 | -1): Matrix {
  return left.map((row, rowIndex) =>
    row.map((value, columnIndex) => value + sign * right[rowIndex][columnIndex]),
  );
}

function cleanResidual(matrix: Matrix): Matrix {
  return matrix.map((row, rowIndex) =>
    row.map((value, columnIndex) => {
      const symmetricValue =
        rowIndex === columnIndex
          ? value
          : (value + matrix[columnIndex][rowIndex]) / 2;

      if (Math.abs(symmetricValue) < TOLERANCE) {
        return 0;
      }

      return rowIndex === columnIndex
        ? Math.max(0, symmetricValue)
        : symmetricValue;
    }),
  );
}

function deterministicRandom(seedText: string): number {
  return mulberry32(hashSeed(seedText))();
}

function hashSeed(seedText: string): number {
  let hash = 1779033703 ^ seedText.length;

  for (let index = 0; index < seedText.length; index += 1) {
    hash = Math.imul(hash ^ seedText.charCodeAt(index), 3432918353);
    hash = (hash << 13) | (hash >>> 19);
  }

  return hash >>> 0;
}

function mulberry32(seed: number): () => number {
  return () => {
    let value = (seed += 0x6d2b79f5);
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
