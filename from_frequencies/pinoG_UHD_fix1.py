import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RelativisticQED:
    """Advanced relativistic QED corrections"""
    
    def __init__(self, max_order: int = 8):
        self.max_order = max_order
        self._setup_feynman_diagrams()
    
    def _setup_feynman_diagrams(self):
        """Initialize Feynman diagram contributions"""
        self.vertex_corrections = {
            4: self._fourth_order_vertex,
            6: self._sixth_order_vertex,
            8: self._eighth_order_vertex
        }
        
    def relativistic_correction(self, alpha: float, pi_est: float) -> float:
        """Calculate full relativistic correction"""
        gamma = 1 / np.sqrt(1 - alpha**2)
        
        correction = 0
        for n in range(2, self.max_order + 1, 2):
            correction += self._order_n_correction(n, alpha, pi_est, gamma)
            
        return correction
    
    def _order_n_correction(self, n: int, alpha: float, pi_est: float, gamma: float) -> float:
        """Calculate nth order relativistic correction"""
        # Kinematic factor
        k_factor = (gamma / pi_est)**n
        
        # Vertex correction
        if n in self.vertex_corrections:
            vertex = self.vertex_corrections[n](alpha, pi_est)
        else:
            vertex = self._estimate_vertex(n, alpha, pi_est)
            
        # Self-energy contribution
        self_energy = self._self_energy_correction(n, alpha, pi_est)
        
        # Vacuum polarization
        vp = self._vacuum_polarization_rel(n, alpha, pi_est, gamma)
        
        return k_factor * (vertex + self_energy + vp)
    
    def _fourth_order_vertex(self, alpha: float, pi_est: float) -> float:
        """Fourth-order vertex correction with exact coefficients"""
        x = alpha / pi_est
        return -0.328478965579193 * x**2 + 1.181241456587183 * x**3
    
    def _sixth_order_vertex(self, alpha: float, pi_est: float) -> float:
        """Sixth-order vertex correction"""
        x = alpha / pi_est
        return -1.4952 * x**3 + (0.7689 - 0.6390 * np.log(x)) * x**4
    
    def _eighth_order_vertex(self, alpha: float, pi_est: float) -> float:
        """Eighth-order vertex correction"""
        x = alpha / pi_est
        return (-2.1770 + 0.8689 * np.log(x) - 0.2011 * np.log(x)**2) * x**4
    
    def _estimate_vertex(self, n: int, alpha: float, pi_est: float) -> float:
        """Estimate higher-order vertex corrections using asymptotic series"""
        x = alpha / pi_est
        leading = (-1)**(n//2) * special.gamma(n-1) * zeta(n-1)
        log_terms = sum(special.binom(n-k-1, k) * np.log(x)**k for k in range(n//2))
        return leading * x**(n-2) * log_terms
    
    def _self_energy_correction(self, n: int, alpha: float, pi_est: float) -> float:
        """Calculate self-energy correction with mass renormalization"""
        x = alpha / pi_est
        
        # Mass renormalization factor
        mass_factor = 1 + 3*x/(4*pi_est) * np.log(x)
        
        # Self-energy contribution
        self_energy = 0
        for k in range(1, n+1):
            coeff = (-1)**(k+1) * special.gamma(k+1/2) / special.gamma(1/2)
            self_energy += coeff * x**k
            
        return mass_factor * self_energy
    
    def _vacuum_polarization_rel(self, n: int, alpha: float, pi_est: float, gamma: float) -> float:
        """Calculate relativistic vacuum polarization"""
        x = alpha / pi_est
        
        # Leading order terms
        vp = (2/3) * x * (1 + x/(2*pi_est))
        
        # Higher order corrections
        for k in range(2, n+1):
            # Include gamma factor for relativistic correction
            vp += (gamma**(k-1) * x**k * 
                  (1 + k*x/(2*pi_est) * np.log(gamma)))
            
        return vp

class VisualizationEngine:
    """Advanced visualization capabilities"""
    
    def __init__(self, theme: str = 'dark'):
        self.theme = theme
        self._setup_style()
        
    def _setup_style(self):
        """Configure visualization style"""
        if self.theme == 'dark':
            plt.style.use('dark_background')
            self.colors = plt.cm.viridis
        else:
            plt.style.use('seaborn-whitegrid')
            self.colors = plt.cm.plasma
            
        sns.set_style("whitegrid")
    
    def create_dashboard(self, results: Dict, uncertainty_results: Dict) -> None:
        """Create comprehensive dashboard of results"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # Pi convergence plot
        self._plot_convergence(fig.add_subplot(gs[0, :2]))
        
        # Uncertainty distribution
        self._plot_uncertainty_dist(fig.add_subplot(gs[0, 2]), uncertainty_results)
        
        # QED corrections
        self._plot_qed_contributions(fig.add_subplot(gs[1, :]))
        
        # Relationship comparison
        self._plot_relationship_comparison(fig.add_subplot(gs[2, :2]), results)
        
        # Error metrics
        self._plot_error_metrics(fig.add_subplot(gs[2, 2]), results)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, results: Dict) -> go.Figure:
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{"type": "scatter"}, {"type": "indicator"}],
                  [{"type": "heatmap"}, {"type": "scatter3d"}],
                  [{"type": "scatter"}, {"type": "bar"}]],
            subplot_titles=("Convergence", "Current Estimate", 
                          "Correlation Matrix", "QED Contributions",
                          "Error Distribution", "Relationship Strengths")
        )
        
        # Add convergence trace
        self._add_convergence_trace(fig, results, row=1, col=1)
        
        # Add current estimate indicator
        self._add_estimate_indicator(fig, results, row=1, col=2)
        
        # Add correlation heatmap
        self._add_correlation_heatmap(fig, results, row=2, col=1)
        
        # Add 3D QED visualization
        self._add_qed_3d_viz(fig, results, row=2, col=2)
        
        # Add error distribution
        self._add_error_distribution(fig, results, row=3, col=1)
        
        # Add relationship strengths
        self._add_relationship_strengths(fig, results, row=3, col=2)
        
        fig.update_layout(height=1200, showlegend=True)
        return fig
    
    def _plot_convergence(self, ax: plt.Axes, history: List[float]) -> None:
        """Plot convergence history with confidence bands"""
        x = np.arange(len(history))
        y = np.array(history)
        
        # Calculate rolling statistics
        window = max(len(history) // 20, 5)
        rolling_mean = pd.Series(y).rolling(window=window).mean()
        rolling_std = pd.Series(y).rolling(window=window).std()
        
        ax.plot(x, y, 'o-', alpha=0.5, label='Estimates')
        ax.plot(x, rolling_mean, 'r-', label='Rolling Mean')
        ax.fill_between(x, rolling_mean - 2*rolling_std, 
                       rolling_mean + 2*rolling_std,
                       alpha=0.2, label='95% CI')
        
        ax.set_title('Convergence History')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('π Estimate')
        ax.legend()
    
    def _plot_qed_contributions(self, ax: plt.Axes, qed_results: Dict) -> None:
        """Create detailed QED contributions plot"""
        orders = list(qed_results.keys())
        contributions = [abs(v) for v in qed_results.values()]
        
        # Create log-scale bar plot
        bars = ax.bar(orders, contributions)
        ax.set_yscale('log')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2e}',
                   ha='center', va='bottom', rotation=0)
            
        ax.set_title('QED Contributions by Order')
        ax.set_xlabel('Order')
        ax.set_ylabel('Absolute Contribution')
    
    def _add_qed_3d_viz(self, fig: go.Figure, results: Dict, row: int, col: int) -> None:
        """Add 3D visualization of QED contributions"""
        orders = np.array(list(results['qed_contributions'].keys()))
        values = np.array(list(results['qed_contributions'].values()))
        uncertainties = np.array(list(results['qed_uncertainties'].values()))
        
        fig.add_trace(
            go.Scatter3d(
                x=orders,
                y=values,
                z=uncertainties,
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=values,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f'Order {o}' for o in orders],
                name='QED Contributions'
            ),
            row=row, col=col
        )
        
        fig.update_layout(scene=dict(
            xaxis_title='Order',
            yaxis_title='Contribution',
            zaxis_title='Uncertainty'
        ))

    @staticmethod
    def generate_report(results: Dict) -> str:
        """Generate detailed analysis report"""
        report = [
            "# Pi Estimation Analysis Report\n",
            f"## Final Estimate: {results['best_pi']:.15f}\n",
            "\n### QED Contributions\n"
        ]
        
        # Add QED analysis
        for order, contrib in results['qed_contributions'].items():
            report.append(f"- Order {order}: {contrib:.2e}")
        
        # Add uncertainty analysis
        report.append("\n### Uncertainty Analysis\n")
        report.append(f"- Standard Error: {results['std_error']:.2e}")
        report.append(f"- 95% CI: [{results['ci_lower']:.15f}, {results['ci_upper']:.15f}]")
        
        return "\n".join(report)

class PiEstimator:  # Final enhanced version
    def __init__(self):
        super().__init__()
        self.rel_qed = RelativisticQED()
        self.visualizer = VisualizationEngine()
        
    def estimate_with_visualization(self) -> Tuple[float, Dict, plt.Figure]:
        """Perform estimation with comprehensive visualization"""
        best_pi, uncertainty_results = self.optimize_pi()
        
        # Generate additional results for visualization
        results = {
            'best_pi': best_pi,
            'convergence_history': self.convergence_analyzer.history,
            'qed_contributions': self.get_qed_contributions(best_pi),
            'relationship_results': self.analyze_relationships(best_pi),
            'uncertainty_results': uncertainty_results
        }
        
        # Create visualization dashboard
        dashboard = self.visualizer.create_dashboard(results, uncertainty_results)
        interactive_dashboard = self.visualizer.create_interactive_dashboard(results)
        
        # Generate report
        report = self.visualizer.generate_report(results)
        
        return best_pi, results, dashboard, interactive_dashboard, report

def main():
    """Enhanced main execution function"""
    estimator = PiEstimator()
    best_pi, results, dashboard, interactive_dashboard, report = estimator.estimate_with_visualization()
    
    # Save visualizations
    dashboard.savefig('pi_estimation_dashboard.png', dpi=300, bbox_inches='tight')
    interactive_dashboard.write_html('pi_estimation_interactive.html')
    
    # Save report
    with open('pi_estimation_report.md', 'w') as f:
        f.write(report)
    
    print("Estimation completed. Results saved to files.")

if __name__ == "__main__":
    main()
