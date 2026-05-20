import { useEffect, useMemo, useState } from "react";
import PlotlyModule from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { DisplayMath, InlineMath } from "./Latex";
import {
  EXAMPLES,
  advanceState,
  choosePivot,
  createInitialState,
  eligiblePivots,
  formatNumber,
  gramMatrix,
  norm,
  orthonormalBasis,
  projectFeatures,
  residualDiagonal,
  residualProbabilities,
  subtract,
  subtractMatrices,
  trace,
  type Dimension,
  type Matrix,
  type PivotState,
  type PivotStrategy,
  type Vector,
} from "./math";

const FEATURE_COLOR = "#2448a6";
const PROJECTED_COLOR = "#0f766e";
const RESIDUAL_COLOR = "#c2410c";
const PIVOT_COLOR = "#b91c1c";
const SUBSPACE_COLOR = "#111827";
const Plot = (
  PlotlyModule as typeof PlotlyModule & {
    default?: typeof PlotlyModule;
  }
).default ?? PlotlyModule;

function App() {
  const [dimension, setDimension] = useState<Dimension>(2);
  const [strategy, setStrategy] = useState<PivotStrategy>("residual");
  const [seed, setSeed] = useState("7");
  const [manualPivot, setManualPivot] = useState("auto");
  const [speed, setSpeed] = useState(5);
  const [featureTextByDimension, setFeatureTextByDimension] = useState<
    Record<Dimension, string>
  >(() => ({
    2: formatFeatureInput(EXAMPLES[2].features),
    3: formatFeatureInput(EXAMPLES[3].features),
  }));
  const [activeFeaturesByDimension, setActiveFeaturesByDimension] = useState<
    Record<Dimension, Matrix>
  >(() => ({
    2: cloneMatrix(EXAMPLES[2].features),
    3: cloneMatrix(EXAMPLES[3].features),
  }));
  const [featureInputError, setFeatureInputError] = useState("");
  const [controlsCollapsed, setControlsCollapsed] = useState(false);
  const [showTargetMatrix, setShowTargetMatrix] = useState(true);
  const [showApproximationMatrix, setShowApproximationMatrix] = useState(true);
  const [showResidualMatrix, setShowResidualMatrix] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [viewRevision, setViewRevision] = useState(0);
  const [history, setHistory] = useState<PivotState[]>(() => [
    createInitialState(EXAMPLES[2].features),
  ]);
  const [historyIndex, setHistoryIndex] = useState(0);

  const features = activeFeaturesByDimension[dimension];
  const featureText = featureTextByDimension[dimension];
  const currentState = history[historyIndex];
  const currentStep = currentState.pivots.length;
  const targetGram = gramMatrix(features);
  const matrixError = subtractMatrices(targetGram, currentState.approximation);
  const matrixScale = Math.max(maxAbsoluteEntry(targetGram), 1);
  const relativeFrobeniusError =
    frobeniusNorm(matrixError) / Math.max(frobeniusNorm(targetGram), 1e-12);
  const currentDiagonal = residualDiagonal(currentState.residual);
  const currentProbabilities = residualProbabilities(currentState.residual);
  const currentTraceResidual = trace(currentState.residual);
  const eligible = eligiblePivots(currentState);
  const basis = orthonormalBasis(features, currentState.pivots);
  const projectedFeatures = projectFeatures(features, basis);
  const residualVectors = features.map((vector, index) =>
    subtract(vector, projectedFeatures[index]),
  );
  const extent = computeExtent(features);
  const animationDelay = 1450 - speed * 125;
  const workspaceClassName = [
    "workspace",
    controlsCollapsed ? "controls-collapsed" : "controls-expanded",
  ].join(" ");

  useEffect(() => {
    if (!isRunning) {
      return;
    }

    if (historyIndex >= history.length - 1 && eligible.length === 0) {
      setIsRunning(false);
      return;
    }

    const timer = window.setTimeout(() => {
      advanceOneStep(false);
    }, animationDelay);

    return () => window.clearTimeout(timer);
  }, [
    animationDelay,
    currentState,
    eligible.length,
    history.length,
    historyIndex,
    isRunning,
    manualPivot,
    seed,
    strategy,
  ]);

  useEffect(() => {
    if (!isFullscreen) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsFullscreen(false);
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isFullscreen]);

  useEffect(() => {
    // Plotly listens to window resizes, but the control panel only changes the grid.
    // Nudge it after the transition so the plot cannot keep an obsolete wider size.
    const animationFrame = window.requestAnimationFrame(() => {
      window.dispatchEvent(new Event("resize"));
    });
    const transitionTimer = window.setTimeout(() => {
      window.dispatchEvent(new Event("resize"));
    }, 240);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      window.clearTimeout(transitionTimer);
    };
  }, [controlsCollapsed]);

  const plotState = useMemo(() => {
    const options: TraceOptions = {
      features,
      projectedFeatures,
      residualVectors,
      pivots: currentState.pivots,
      basis,
      extent,
    };

    if (dimension === 2) {
      return {
        data: create2DTraces(options),
        layout: create2DLayout(options, viewRevision),
      };
    }

    return {
      data: create3DTraces(options),
      layout: create3DLayout(options, viewRevision),
    };
  }, [
    basis,
    currentState.pivots,
    dimension,
    extent,
    features,
    projectedFeatures,
    residualVectors,
    viewRevision,
  ]);
  const heatmapState = useMemo(
    () => ({
      target: createHeatmapState(targetGram, matrixScale, "gram"),
      approximation: createHeatmapState(
        currentState.approximation,
        matrixScale,
        "gram",
      ),
      residual: createHeatmapState(currentState.residual, matrixScale, "gram"),
    }),
    [currentState.approximation, currentState.residual, matrixScale, targetGram],
  );
  const matrixOverlays = [
    showTargetMatrix
      ? {
          key: "target",
          title: String.raw`A`,
          state: heatmapState.target,
        }
      : null,
    showApproximationMatrix
      ? {
          key: "approximation",
          title: String.raw`\widehat A_${currentStep}`,
          state: heatmapState.approximation,
        }
      : null,
    showResidualMatrix
      ? {
          key: "residual",
          title: String.raw`R_${currentStep}`,
          state: heatmapState.residual,
        }
      : null,
  ].filter((overlay): overlay is MatrixOverlay => overlay !== null);

  function resetAlgorithm() {
    resetAlgorithmWithFeatures(features);
  }

  function resetAlgorithmWithFeatures(nextFeatures: Matrix) {
    setHistory([createInitialState(nextFeatures)]);
    setHistoryIndex(0);
    setManualPivot("auto");
    setIsRunning(false);
  }

  function handleDimensionChange(nextDimension: Dimension) {
    setDimension(nextDimension);
    resetAlgorithmWithFeatures(activeFeaturesByDimension[nextDimension]);
    setFeatureInputError("");
  }

  function advanceOneStep(pauseAfter: boolean) {
    if (pauseAfter) {
      setIsRunning(false);
    }

    const selectedManualPivot =
      manualPivot === "auto" ? null : Number(manualPivot);

    if (historyIndex < history.length - 1 && selectedManualPivot === null) {
      setHistoryIndex((index) => index + 1);
      return;
    }

    const pivot = choosePivot(
      currentState,
      strategy,
      seed,
      selectedManualPivot,
    );

    if (pivot === null) {
      setIsRunning(false);
      return;
    }

    const nextState = advanceState(currentState, pivot);
    if (!nextState) {
      setIsRunning(false);
      return;
    }

    setHistory((states) => [...states.slice(0, historyIndex + 1), nextState]);
    setHistoryIndex((index) => index + 1);
    setManualPivot("auto");
  }

  function stepBack() {
    setHistoryIndex((index) => Math.max(0, index - 1));
    setIsRunning(false);
  }

  function goToStep(targetStep: number) {
    setIsRunning(false);

    if (targetStep <= history.length - 1) {
      setHistoryIndex(targetStep);
      return;
    }

    const nextHistory = [...history];

    while (nextHistory.length - 1 < targetStep) {
      const state = nextHistory[nextHistory.length - 1];
      const pivot = choosePivot(state, strategy, seed, null);
      const nextState = pivot === null ? null : advanceState(state, pivot);

      if (!nextState) {
        break;
      }

      nextHistory.push(nextState);
    }

    setHistory(nextHistory);
    setHistoryIndex(Math.min(targetStep, nextHistory.length - 1));
    setManualPivot("auto");
  }

  function resetView() {
    setViewRevision((revision) => revision + 1);
  }

  function handleStrategyChange(nextStrategy: PivotStrategy) {
    setStrategy(nextStrategy);
    setHistory((states) => states.slice(0, historyIndex + 1));
    setIsRunning(false);
  }

  function handleSeedChange(nextSeed: string) {
    setSeed(nextSeed);
    setHistory((states) => states.slice(0, historyIndex + 1));
    setIsRunning(false);
  }

  function handleFeatureTextChange(nextText: string) {
    setFeatureTextByDimension((texts) => ({
      ...texts,
      [dimension]: nextText,
    }));
    setFeatureInputError("");
  }

  function applyCustomFeatures() {
    const parsed = parseFeatureMatrix(featureText, dimension);

    if ("error" in parsed) {
      setFeatureInputError(parsed.error);
      return;
    }

    setActiveFeaturesByDimension((featureSets) => ({
      ...featureSets,
      [dimension]: parsed.features,
    }));
    resetAlgorithmWithFeatures(parsed.features);
    setFeatureInputError("");
  }

  function restorePresetFeatures() {
    const preset = cloneMatrix(EXAMPLES[dimension].features);

    setActiveFeaturesByDimension((featureSets) => ({
      ...featureSets,
      [dimension]: preset,
    }));
    setFeatureTextByDimension((texts) => ({
      ...texts,
      [dimension]: formatFeatureInput(preset),
    }));
    resetAlgorithmWithFeatures(preset);
    setFeatureInputError("");
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Interactive seminar demo</p>
          <h1>Randomly Pivoted Cholesky</h1>
        </div>
        <p className="hero-copy">
          Watch pivoted Cholesky build a Nyström approximation by projecting
          hidden feature vectors onto a growing subspace.
        </p>
      </header>

      <section className={workspaceClassName}>
        <div
          className={`control-shell ${controlsCollapsed ? "is-collapsed" : ""}`}
        >
          <aside className="control-panel" aria-hidden={controlsCollapsed}>
            <button
              type="button"
              className="control-panel-toggle"
              onClick={() => setControlsCollapsed(true)}
              aria-label="Collapse controls"
            >
              {"<<"}
            </button>

            <div className="panel-block">
              <label>
                Dimension
                <select
                  value={dimension}
                  onChange={(event) =>
                    handleDimensionChange(
                      Number(event.target.value) as Dimension,
                    )
                  }
                >
                  <option value={2}>2D example</option>
                  <option value={3}>3D example</option>
                </select>
              </label>

              <label>
                Pivot strategy
                <select
                  value={strategy}
                  onChange={(event) =>
                    handleStrategyChange(event.target.value as PivotStrategy)
                  }
                >
                  <option value="residual">Random residual pivoting</option>
                  <option value="greedy">Greedy pivoting</option>
                  <option value="uniform">Uniform random pivoting</option>
                </select>
              </label>

              <label>
                Manual next pivot
                <select
                  value={manualPivot}
                  onChange={(event) => setManualPivot(event.target.value)}
                  disabled={eligible.length === 0}
                >
                  <option value="auto">Automatic</option>
                  {eligible.map((index) => (
                    <option key={index} value={index}>
                      v{index + 1}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Random seed
                <input
                  type="text"
                  value={seed}
                  onChange={(event) => handleSeedChange(event.target.value)}
                />
              </label>
            </div>

            <div className="panel-block">
              <div className="button-row">
                <button
                  type="button"
                  onClick={() => setIsRunning(true)}
                  disabled={isRunning || eligible.length === 0}
                >
                  Run animation
                </button>
                <button
                  type="button"
                  onClick={() => setIsRunning(false)}
                  disabled={!isRunning}
                >
                  Pause
                </button>
                <button type="button" onClick={resetAlgorithm}>
                  Reset
                </button>
                <button
                  type="button"
                  onClick={stepBack}
                  disabled={historyIndex === 0}
                >
                  Previous step
                </button>
                <button
                  type="button"
                  onClick={() => advanceOneStep(true)}
                  disabled={eligible.length === 0 && historyIndex >= history.length - 1}
                >
                  Next pivot
                </button>
              </div>

              <label>
                Animation speed
                <div className="range-line">
                  <input
                    type="range"
                    min={1}
                    max={10}
                    value={speed}
                    onChange={(event) => setSpeed(Number(event.target.value))}
                  />
                  <span>{speed}/10</span>
                </div>
              </label>

              <label>
                Iteration / rank
                <div className="range-line">
                  <input
                    type="range"
                    min={0}
                    max={features.length}
                    step={1}
                    value={historyIndex}
                    onChange={(event) => goToStep(Number(event.target.value))}
                  />
                  <span>
                    <InlineMath>{String.raw`t=${currentStep}`}</InlineMath>
                  </span>
                </div>
              </label>
            </div>

            <div className="panel-block">
              <h2>Matrices in plot</h2>
              <div className="matrix-toggle-list">
                <label>
                  <input
                    type="checkbox"
                    checked={showTargetMatrix}
                    onChange={(event) =>
                      setShowTargetMatrix(event.target.checked)
                    }
                  />
                  <InlineMath>A</InlineMath>
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={showApproximationMatrix}
                    onChange={(event) =>
                      setShowApproximationMatrix(event.target.checked)
                    }
                  />
                  <InlineMath>{String.raw`\widehat A_t`}</InlineMath>
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={showResidualMatrix}
                    onChange={(event) =>
                      setShowResidualMatrix(event.target.checked)
                    }
                  />
                  <InlineMath>{String.raw`R_t`}</InlineMath>
                </label>
              </div>
            </div>

            <div className="panel-block data-panel">
              <h2>Custom feature vectors</h2>
              <label>
                Feature vectors
                <textarea
                  value={featureText}
                  onChange={(event) =>
                    handleFeatureTextChange(event.target.value)
                  }
                  spellCheck={false}
                  rows={dimension === 2 ? 7 : 8}
                  aria-label="Custom feature vectors"
                />
              </label>

              {featureInputError && (
                <p className="input-error" role="alert">
                  {featureInputError}
                </p>
              )}

              <div className="compact-button-row">
                <button type="button" onClick={applyCustomFeatures}>
                  Apply vectors
                </button>
                <button type="button" onClick={restorePresetFeatures}>
                  Use preset
                </button>
              </div>
            </div>
          </aside>

          <button
            type="button"
            className="control-reopen-button"
            onClick={() => setControlsCollapsed(false)}
            aria-label="Expand controls"
          >
            {">>"}
          </button>
        </div>

        <section className="visual-panel">
          <div className="visual-toolbar">
            <div className="toolbar-actions">
              <button type="button" onClick={() => setIsFullscreen(true)}>
                Fullscreen plot
              </button>
              <button
                type="button"
                onClick={stepBack}
                disabled={historyIndex === 0}
              >
                Previous step
              </button>
              <button
                type="button"
                onClick={() => advanceOneStep(true)}
                disabled={eligible.length === 0 && historyIndex >= history.length - 1}
              >
                Next pivot
              </button>
              <button type="button" onClick={resetView}>
                Reset view
              </button>
            </div>
          </div>

          <div className="plot-stage">
            <Plot
              data={plotState.data}
              layout={plotState.layout}
              config={{
                displaylogo: false,
                responsive: true,
                scrollZoom: true,
              }}
              useResizeHandler
              className="plot"
            />
            <div className="pivot-overlay">
              <DisplayMath>{formatSubspaceLatex(currentState.pivots)}</DisplayMath>
            </div>
            <div className="matrix-overlay-stack">
              {matrixOverlays.map((overlay) => (
                <article key={overlay.key} className="matrix-overlay-card">
                  <h3>
                    <InlineMath>{overlay.title}</InlineMath>
                  </h3>
                  <Plot
                    data={overlay.state.data}
                    layout={overlay.state.layout}
                    config={{ displaylogo: false, responsive: true }}
                    useResizeHandler
                    className="matrix-overlay-plot"
                  />
                </article>
              ))}
            </div>
          </div>
        </section>

        <aside className="readout-panel readout-detail-panel">
            <details className="panel-block collapsible-block formula-panel" open>
              <summary className="section-summary">
                <span>Current formulas</span>
                <span className="section-chevron" aria-hidden="true" />
              </summary>
              <div className="section-content">
                <div className="formula-entry">
                  <strong>Projected Gram approximation</strong>
                  <DisplayMath>
                    {String.raw`\widehat A_${currentStep}(i,j)=\langle P_{U_${currentStep}}v_i,\;P_{U_${currentStep}}v_j\rangle`}
                  </DisplayMath>
                </div>
                <div className="formula-entry">
                  <strong>Residual matrix</strong>
                  <DisplayMath>
                    {String.raw`R_${currentStep}=A-\widehat A_${currentStep}`}
                  </DisplayMath>
                </div>
                <div className="formula-entry">
                  <strong>Residual diagonal</strong>
                  <DisplayMath>
                    {String.raw`R_${currentStep}(j,j)=\lVert(I-P_{U_${currentStep}})v_j\rVert^2`}
                  </DisplayMath>
                </div>
                <div className="formula-entry">
                  <strong>Random pivot rule</strong>
                  <DisplayMath>
                    {String.raw`\mathbb{P}(s_${currentStep + 1}=j)=\frac{R_${currentStep}(j,j)}{\operatorname{tr}(R_${currentStep})}`}
                  </DisplayMath>
                </div>
                <div className="formula-entry">
                  <strong>Rank-one update</strong>
                  <DisplayMath>
                    {String.raw`\widehat A_${currentStep + 1}=\widehat A_${currentStep}+\frac{R_${currentStep}(:,s_${currentStep + 1})R_${currentStep}(s_${currentStep + 1},:)}{R_${currentStep}(s_${currentStep + 1},s_${currentStep + 1})}`}
                  </DisplayMath>
                </div>
                <div className="formula-entry">
                  <strong>Residual update</strong>
                  <DisplayMath>
                    {String.raw`R_${currentStep + 1}=R_${currentStep}-\frac{R_${currentStep}(:,s_${currentStep + 1})R_${currentStep}(s_${currentStep + 1},:)}{R_${currentStep}(s_${currentStep + 1},s_${currentStep + 1})}`}
                  </DisplayMath>
                </div>
              </div>
            </details>

            <details className="panel-block collapsible-block" open>
              <summary className="section-summary">
                <span>Error measures</span>
                <span className="section-chevron" aria-hidden="true" />
              </summary>
              <div className="section-content">
                <dl>
                  <div>
                    <dt>Total unexplained mass</dt>
                    <dd>
                      <InlineMath>
                        {String.raw`\operatorname{tr}(R_${currentStep})=${formatNumber(
                          currentTraceResidual,
                        )}`}
                      </InlineMath>
                    </dd>
                  </div>
                  <div>
                    <dt>Relative Frobenius error</dt>
                    <dd>
                      <InlineMath>
                        {String.raw`\frac{\lVert R_${currentStep}\rVert_F}{\lVert A\rVert_F}=${formatNumber(
                          relativeFrobeniusError,
                        )}`}
                      </InlineMath>
                    </dd>
                  </div>
                </dl>
              </div>
            </details>

            <details className="panel-block collapsible-block" open>
              <summary className="section-summary">
                <span>Residual diagonal</span>
                <span className="section-chevron" aria-hidden="true" />
              </summary>
              <div className="section-content">
                <ol className="value-list">
                  {currentDiagonal.map((value, index) => (
                    <li key={index}>
                      <InlineMath>
                        {String.raw`R_${currentStep}(${index + 1},${index + 1})=${formatNumber(
                          value,
                        )}`}
                      </InlineMath>
                    </li>
                  ))}
                </ol>
              </div>
            </details>

            <details className="panel-block collapsible-block" open>
              <summary className="section-summary">
                <span>Sampling probabilities</span>
                <span className="section-chevron" aria-hidden="true" />
              </summary>
              <div className="section-content">
                <ol className="value-list">
                  {currentProbabilities.map((value, index) => (
                    <li key={index}>
                      <InlineMath>
                        {String.raw`p_${index + 1}=${formatNumber(value)}`}
                      </InlineMath>
                    </li>
                  ))}
                </ol>
              </div>
            </details>

            <details className="panel-block collapsible-block" open>
              <summary className="section-summary">
                <span>Projected vectors</span>
                <span className="section-chevron" aria-hidden="true" />
              </summary>
              <div className="section-content">
                <ol className="value-list projected-list">
                  {projectedFeatures.map((vector, index) => (
                    <li key={index}>
                      <InlineMath>
                        {String.raw`\tilde v_${index + 1}=${formatVectorLatex(vector)}`}
                      </InlineMath>
                    </li>
                  ))}
                </ol>
              </div>
            </details>

          </aside>
      </section>

      {isFullscreen && (
        <div
          className="fullscreen-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label="Fullscreen feature-space plot"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setIsFullscreen(false);
            }
          }}
        >
          <section className="fullscreen-panel">
            <div className="fullscreen-toolbar">
              <div className="fullscreen-controls">
                <button
                  type="button"
                  onClick={() => setIsRunning(true)}
                  disabled={isRunning || eligible.length === 0}
                >
                  Run
                </button>
                <button
                  type="button"
                  onClick={() => setIsRunning(false)}
                  disabled={!isRunning}
                >
                  Pause
                </button>
                <button type="button" onClick={resetAlgorithm}>
                  Reset
                </button>
                <button
                  type="button"
                  onClick={stepBack}
                  disabled={historyIndex === 0}
                >
                  Previous
                </button>
                <button
                  type="button"
                  onClick={() => advanceOneStep(true)}
                  disabled={eligible.length === 0 && historyIndex >= history.length - 1}
                >
                  Next pivot
                </button>
                <button type="button" onClick={resetView}>
                  Reset view
                </button>
              </div>

              <button
                type="button"
                className="close-button"
                onClick={() => setIsFullscreen(false)}
              >
                Close
              </button>
            </div>

            <div className="fullscreen-plot-stage">
              <Plot
                data={plotState.data}
                layout={plotState.layout}
                config={{
                  displaylogo: false,
                  responsive: true,
                  scrollZoom: true,
                }}
                useResizeHandler
                className="plot fullscreen-plot"
              />
              <div className="pivot-overlay">
                <DisplayMath>{formatSubspaceLatex(currentState.pivots)}</DisplayMath>
              </div>
              <div className="matrix-overlay-stack">
                {matrixOverlays.map((overlay) => (
                  <article key={overlay.key} className="matrix-overlay-card">
                    <h3>
                      <InlineMath>{overlay.title}</InlineMath>
                    </h3>
                    <Plot
                      data={overlay.state.data}
                      layout={overlay.state.layout}
                      config={{ displaylogo: false, responsive: true }}
                      useResizeHandler
                      className="matrix-overlay-plot"
                    />
                  </article>
                ))}
              </div>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

type TraceOptions = {
  features: Matrix;
  projectedFeatures: Matrix;
  residualVectors: Matrix;
  pivots: number[];
  basis: Matrix;
  extent: number;
};

type MatrixOverlay = {
  key: string;
  title: string;
  state: { data: Data[]; layout: Partial<Layout> };
};

function create2DTraces(options: TraceOptions): Data[] {
  const traces: Data[] = [
    {
      type: "scatter",
      mode: "lines",
      ...createSegmentSeries(options.features),
      name: "original vectors",
      hoverinfo: "skip",
      line: { color: FEATURE_COLOR, width: 3 },
    },
    {
      type: "scatter",
      mode: "text+markers",
      x: options.features.map((vector) => vector[0]),
      y: options.features.map((vector) => vector[1]),
      text: options.features.map((_, index) => `v<sub>${index + 1}</sub>`),
      textposition: "top center",
      textfont: { size: 16 },
      name: "feature vectors",
      marker: {
        color: FEATURE_COLOR,
        size: 13,
      },
      hovertemplate: options.features.map(
        (vector, index) =>
          `${hoverVector(`v<sub>${index + 1}</sub>`, vector)}<extra></extra>`,
      ),
    },
    {
      type: "scatter",
      mode: "lines",
      ...createResidualSeries(options.projectedFeatures, options.features),
      name: "residual arrows",
      hoverinfo: "skip",
      line: { color: RESIDUAL_COLOR, width: 3, dash: "dash" },
    },
  ];

  if (options.basis.length > 0) {
    traces.push(
      {
        type: "scatter",
        mode: "lines",
        ...createSegmentSeries(options.projectedFeatures),
        name: "projected vectors",
        hoverinfo: "skip",
        line: { color: PROJECTED_COLOR, width: 3 },
      },
      {
        type: "scatter",
        mode: "text+markers",
        x: options.projectedFeatures.map((vector) => vector[0]),
        y: options.projectedFeatures.map((vector) => vector[1]),
        text: options.projectedFeatures.map(
          (_, index) => `ṽ<sub>${index + 1}</sub>`,
        ),
        textposition: "bottom center",
        textfont: { size: 16 },
        name: "projected points",
        marker: {
          color: PROJECTED_COLOR,
          size: 12,
          symbol: "square",
        },
      },
    );
  }

  if (options.pivots.length > 0) {
    traces.push({
      type: "scatter",
      mode: "markers",
      x: options.pivots.map((index) => options.features[index][0]),
      y: options.pivots.map((index) => options.features[index][1]),
      name: "selected pivots",
      marker: {
        color: PIVOT_COLOR,
        size: 17,
        symbol: "star",
      },
    });
  }

  if (options.basis.length === 1) {
    const basisVector = options.basis[0];
    traces.push({
      type: "scatter",
      mode: "lines",
      x: [-basisVector[0] * options.extent, basisVector[0] * options.extent],
      y: [-basisVector[1] * options.extent, basisVector[1] * options.extent],
      name: "U<sub>t</sub>",
      line: { color: SUBSPACE_COLOR, width: 5 },
    });
  }

  return traces;
}

function create3DTraces(options: TraceOptions): Data[] {
  const traces: Data[] = [
    {
      type: "scatter3d",
      mode: "lines",
      ...createSegmentSeries3D(options.features),
      name: "original vectors",
      hoverinfo: "skip",
      line: { color: FEATURE_COLOR, width: 6 },
    },
    {
      type: "scatter3d",
      mode: "text+markers",
      x: options.features.map((vector) => vector[0]),
      y: options.features.map((vector) => vector[1]),
      z: options.features.map((vector) => vector[2]),
      text: options.features.map((_, index) => `v<sub>${index + 1}</sub>`),
      textposition: "top center",
      textfont: { size: 15 },
      name: "feature vectors",
      marker: {
        color: FEATURE_COLOR,
        size: 7,
      },
    },
    {
      type: "scatter3d",
      mode: "lines",
      ...createResidualSeries3D(options.projectedFeatures, options.features),
      name: "residual arrows",
      hoverinfo: "skip",
      line: { color: RESIDUAL_COLOR, width: 6, dash: "dash" },
    },
  ];

  if (options.basis.length > 0) {
    traces.push(
      {
        type: "scatter3d",
        mode: "lines",
        ...createSegmentSeries3D(options.projectedFeatures),
        name: "projected vectors",
        hoverinfo: "skip",
        line: { color: PROJECTED_COLOR, width: 6 },
      },
      {
        type: "scatter3d",
        mode: "text+markers",
        x: options.projectedFeatures.map((vector) => vector[0]),
        y: options.projectedFeatures.map((vector) => vector[1]),
        z: options.projectedFeatures.map((vector) => vector[2]),
        text: options.projectedFeatures.map(
          (_, index) => `ṽ<sub>${index + 1}</sub>`,
        ),
        textposition: "bottom center",
        textfont: { size: 15 },
        name: "projected points",
        marker: {
          color: PROJECTED_COLOR,
          size: 7,
          symbol: "square",
        },
      },
    );
  }

  if (options.pivots.length > 0) {
    traces.push({
      type: "scatter3d",
      mode: "markers",
      x: options.pivots.map((index) => options.features[index][0]),
      y: options.pivots.map((index) => options.features[index][1]),
      z: options.pivots.map((index) => options.features[index][2]),
      name: "selected pivots",
      marker: {
        color: PIVOT_COLOR,
        size: 8,
        symbol: "diamond",
      },
    });
  }

  if (options.basis.length === 1) {
    const basisVector = options.basis[0];
    traces.push({
      type: "scatter3d",
      mode: "lines",
      x: [-basisVector[0] * options.extent, basisVector[0] * options.extent],
      y: [-basisVector[1] * options.extent, basisVector[1] * options.extent],
      z: [-basisVector[2] * options.extent, basisVector[2] * options.extent],
      name: "U<sub>t</sub>",
      line: { color: SUBSPACE_COLOR, width: 8 },
    });
  }

  if (options.basis.length === 2) {
    traces.push(createSubspacePlane(options.basis, options.extent));
  }

  const nonZeroResiduals = options.residualVectors
    .map((vector, index) => ({ vector, index }))
    .filter(({ vector }) => norm(vector) > 1e-8);

  if (nonZeroResiduals.length > 0) {
    traces.push({
      type: "cone",
      x: nonZeroResiduals.map(({ index }) => options.features[index][0]),
      y: nonZeroResiduals.map(({ index }) => options.features[index][1]),
      z: nonZeroResiduals.map(({ index }) => options.features[index][2]),
      u: nonZeroResiduals.map(({ vector }) => vector[0]),
      v: nonZeroResiduals.map(({ vector }) => vector[1]),
      w: nonZeroResiduals.map(({ vector }) => vector[2]),
      anchor: "tip",
      sizemode: "absolute",
      sizeref: 0.22,
      showscale: false,
      showlegend: false,
      hoverinfo: "skip",
      opacity: 0.8,
      colorscale: [
        [0, RESIDUAL_COLOR],
        [1, RESIDUAL_COLOR],
      ],
    } as unknown as Data);
  }

  return traces;
}

function create2DLayout(
  options: TraceOptions,
  viewRevision: number,
): Partial<Layout> {
  return {
    autosize: true,
    margin: { l: 52, r: 20, t: 28, b: 48 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    hovermode: "closest",
    dragmode: "pan",
    legend: {
      orientation: "h",
      y: 1.12,
      x: 0,
    },
    annotations: options.features
      .map((feature, index) => ({
        x: feature[0],
        y: feature[1],
        ax: options.projectedFeatures[index][0],
        ay: options.projectedFeatures[index][1],
        xref: "x" as const,
        yref: "y" as const,
        axref: "x" as const,
        ayref: "y" as const,
        showarrow: norm(options.residualVectors[index]) > 1e-8,
        arrowhead: 3,
        arrowsize: 1.1,
        arrowwidth: 2,
        arrowcolor: RESIDUAL_COLOR,
      }))
      .filter((annotation) => annotation.showarrow),
    xaxis: {
      title: { text: "feature coordinate 1" },
      range: [-options.extent, options.extent],
      zeroline: true,
      gridcolor: "#e5e7eb",
      scaleanchor: "y",
      scaleratio: 1,
    },
    yaxis: {
      title: { text: "feature coordinate 2" },
      range: [-options.extent, options.extent],
      zeroline: true,
      gridcolor: "#e5e7eb",
    },
    uirevision: `cholesky-2d-${viewRevision}`,
  };
}

function create3DLayout(
  options: TraceOptions,
  viewRevision: number,
): Partial<Layout> {
  return {
    autosize: true,
    margin: { l: 0, r: 0, t: 24, b: 0 },
    paper_bgcolor: "#ffffff",
    hovermode: "closest",
    legend: {
      orientation: "h",
      y: 1.1,
      x: 0,
    },
    scene: {
      aspectmode: "cube",
      xaxis: {
        title: { text: "coordinate 1" },
        range: [-options.extent, options.extent],
        gridcolor: "#e5e7eb",
      },
      yaxis: {
        title: { text: "coordinate 2" },
        range: [-options.extent, options.extent],
        gridcolor: "#e5e7eb",
      },
      zaxis: {
        title: { text: "coordinate 3" },
        range: [-options.extent, options.extent],
        gridcolor: "#e5e7eb",
      },
    },
    uirevision: `cholesky-3d-${viewRevision}`,
  };
}

function createSubspacePlane(basis: Matrix, extent: number): Data {
  const first = basis[0];
  const second = basis[1];
  const coordinates = [-extent, extent];

  return {
    type: "surface",
    x: coordinates.map((firstScale) =>
      coordinates.map(
        (secondScale) => firstScale * first[0] + secondScale * second[0],
      ),
    ),
    y: coordinates.map((firstScale) =>
      coordinates.map(
        (secondScale) => firstScale * first[1] + secondScale * second[1],
      ),
    ),
    z: coordinates.map((firstScale) =>
      coordinates.map(
        (secondScale) => firstScale * first[2] + secondScale * second[2],
      ),
    ),
    name: "U<sub>t</sub>",
    opacity: 0.24,
    showscale: false,
    colorscale: [
      [0, SUBSPACE_COLOR],
      [1, SUBSPACE_COLOR],
    ],
  };
}

function createSegmentSeries(vectors: Matrix) {
  return {
    x: vectors.flatMap((vector) => [0, vector[0], null]),
    y: vectors.flatMap((vector) => [0, vector[1], null]),
  };
}

function createSegmentSeries3D(vectors: Matrix) {
  return {
    x: vectors.flatMap((vector) => [0, vector[0], null]),
    y: vectors.flatMap((vector) => [0, vector[1], null]),
    z: vectors.flatMap((vector) => [0, vector[2], null]),
  };
}

function createResidualSeries(projected: Matrix, original: Matrix) {
  return {
    x: original.flatMap((vector, index) => [
      projected[index][0],
      vector[0],
      null,
    ]),
    y: original.flatMap((vector, index) => [
      projected[index][1],
      vector[1],
      null,
    ]),
  };
}

function createResidualSeries3D(projected: Matrix, original: Matrix) {
  return {
    x: original.flatMap((vector, index) => [
      projected[index][0],
      vector[0],
      null,
    ]),
    y: original.flatMap((vector, index) => [
      projected[index][1],
      vector[1],
      null,
    ]),
    z: original.flatMap((vector, index) => [
      projected[index][2],
      vector[2],
      null,
    ]),
  };
}

function computeExtent(features: Matrix): number {
  const largestCoordinate = features
    .flat()
    .reduce((largest, value) => Math.max(largest, Math.abs(value)), 0);

  return Math.max(2.8, Math.ceil(largestCoordinate + 0.6));
}

function createHeatmapState(
  matrix: Matrix,
  scale: number,
  kind: "gram",
): { data: Data[]; layout: Partial<Layout> } {
  const labels = matrix.map((_, index) => `v${index + 1}`);

  return {
    data: [
      {
        type: "heatmap",
        z: matrix,
        x: labels,
        y: labels,
        zmin: -scale,
        zmax: scale,
        zmid: 0,
        colorscale: [
          [0, "#1d4ed8"],
          [0.5, "#f8fafc"],
          [1, "#b91c1c"],
        ],
        xgap: 1,
        ygap: 1,
        showscale: false,
        hovertemplate:
          "row %{y}, column %{x}<br>value %{z:.4f}<extra></extra>",
      } as Data,
    ],
    layout: {
      autosize: true,
      margin: { l: 24, r: 8, t: 4, b: 22 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      xaxis: {
        side: "bottom",
        tickmode: "array",
        tickvals: labels,
        ticktext: labels,
        tickfont: { size: 9 },
      },
      yaxis: {
        autorange: "reversed",
        tickmode: "array",
        tickvals: labels,
        ticktext: labels,
        tickfont: { size: 9 },
        scaleanchor: "x",
      },
    },
  };
}

function maxAbsoluteEntry(matrix: Matrix): number {
  return matrix.reduce(
    (largest, row) =>
      row.reduce((rowLargest, value) => Math.max(rowLargest, Math.abs(value)), largest),
    0,
  );
}

function frobeniusNorm(matrix: Matrix): number {
  return Math.sqrt(
    matrix.reduce(
      (sum, row) =>
        sum + row.reduce((rowSum, value) => rowSum + value * value, 0),
      0,
    ),
  );
}

function formatSubspaceLatex(pivots: number[]): string {
  if (pivots.length === 0) {
    return `U_0=\\{0\\}`;
  }

  return `U_${pivots.length}=\\operatorname{span}\\{${pivots
    .map((pivot) => `v_${pivot + 1}`)
    .join(",")}\\}`;
}

function formatVectorLatex(vector: Vector): string {
  return `\\begin{bmatrix}${vector.map(formatNumber).join(" \\\\ ")}\\end{bmatrix}`;
}

function formatFeatureInput(features: Matrix): string {
  return `[\n${features
    .map((vector) => `  [${vector.map(formatNumber).join(", ")}]`)
    .join(",\n")}\n]`;
}

function cloneMatrix(matrix: Matrix): Matrix {
  return matrix.map((row) => [...row]);
}

function parseFeatureMatrix(
  text: string,
  dimension: Dimension,
): { features: Matrix } | { error: string } {
  let parsed: unknown;

  try {
    parsed = JSON.parse(text);
  } catch {
    return {
      error:
        "Please enter vectors as a JSON-style array, for example [[1, 0], [0, 1]].",
    };
  }

  if (!Array.isArray(parsed) || parsed.length === 0) {
    return { error: "Please provide at least one feature vector." };
  }

  const features: Matrix = [];

  for (const [index, row] of parsed.entries()) {
    if (!Array.isArray(row) || row.length !== dimension) {
      return {
        error: `Vector v${index + 1} must contain exactly ${dimension} entries.`,
      };
    }

    if (!row.every((value) => typeof value === "number" && Number.isFinite(value))) {
      return {
        error: `Vector v${index + 1} must contain only finite numbers.`,
      };
    }

    features.push(row.map((value) => Number(value)));
  }

  return { features };
}

function hoverVector(label: string, vector: Vector): string {
  return `${label} = [${vector.map(formatNumber).join(", ")}]`;
}

export default App;
