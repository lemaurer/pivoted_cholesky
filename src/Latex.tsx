import katex from "katex";
import "katex/dist/katex.min.css";

type MathProps = {
  children: string;
  className?: string;
};

function renderMath(math: string, displayMode: boolean) {
  return katex.renderToString(math, {
    displayMode,
    throwOnError: false,
  });
}

export function InlineMath({ children, className }: MathProps) {
  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: renderMath(children, false) }}
    />
  );
}

export function DisplayMath({ children, className }: MathProps) {
  return (
    <div
      className={className}
      dangerouslySetInnerHTML={{ __html: renderMath(children, true) }}
    />
  );
}
