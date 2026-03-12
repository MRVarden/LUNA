// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import type { SynthesisSnapshot } from '../../api/types'

const METRIC_LABELS: Record<string, string> = {
  psi_0: 'Perception',
  psi_1: 'Reflexion',
  psi_2: 'Integration',
  psi_3: 'Expression',
  phi_iit: 'Phi_IIT',
  thinker_confidence: 'Confiance',
  affect_valence: 'Valence',
  affect_arousal: 'Arousal',
}

function label(metric: string): string {
  return METRIC_LABELS[metric] ?? metric
}

interface Props {
  synthesis: SynthesisSnapshot | null
}

export function SynthesisPanel({ synthesis }: Props) {
  if (!synthesis) {
    return <p className="text-[10px] text-luna-text-muted italic">Aucune synthese disponible.</p>
  }

  const significantTrends = synthesis.trends.filter(
    t => t.r_squared > 0.236 && t.direction !== 'stable'
  )

  return (
    <div className="flex flex-col gap-3 text-[10px]">
      {/* Cycles analyzed */}
      <div className="text-luna-text-muted">
        {synthesis.cycles_analyzed} cycles analyses
      </div>

      {/* Trends */}
      {significantTrends.length > 0 && (
        <div>
          <div className="text-luna-text-dim uppercase tracking-wider mb-1">Tendances</div>
          <div className="flex flex-wrap gap-1.5">
            {significantTrends.map((t, i) => (
              <span
                key={i}
                className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-mono ${
                  t.direction === 'up'
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'bg-red-500/10 text-red-400'
                }`}
              >
                {t.direction === 'up' ? '\u2191' : '\u2193'}
                {label(t.metric)}
                <span className="text-luna-text-muted">R\u00B2={t.r_squared.toFixed(2)}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Anomalies */}
      {synthesis.anomalies.length > 0 && (
        <div>
          <div className="text-luna-text-dim uppercase tracking-wider mb-1">Anomalies</div>
          <div className="flex flex-wrap gap-1.5">
            {synthesis.anomalies.slice(0, 5).map((a, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 text-[9px] font-mono"
              >
                ! {label(a.metric)} #{a.cycle_index}
                <span className="text-luna-text-muted">{a.sigma_deviation.toFixed(1)}&sigma;</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Cross-patterns */}
      {synthesis.cross_patterns.length > 0 && (
        <div>
          <div className="text-luna-text-dim uppercase tracking-wider mb-1">Correlations</div>
          <div className="flex flex-col gap-1">
            {synthesis.cross_patterns.map((cp, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className={`font-mono text-[9px] ${
                  cp.correlation > 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {cp.correlation > 0 ? '+' : ''}{cp.correlation.toFixed(3)}
                </span>
                <span className="text-luna-text-muted">
                  {label(cp.metric_a)} &harr; {label(cp.metric_b)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No findings */}
      {significantTrends.length === 0 && synthesis.anomalies.length === 0 && synthesis.cross_patterns.length === 0 && (
        <p className="text-luna-text-muted italic">Aucune tendance significative.</p>
      )}
    </div>
  )
}
