\begin{table}[!h]

\caption{\label{tab:}Shows the sucess-rate for extrapolation < $\epsilon$, at what global step the model converged at, and the sparse error for all weight matrices.}
\centering
\begin{tabular}{rrlll}
\toprule
operation & model & success.rate & converged.at & sparse.error\\
\midrule
 & ${\mathrm{NAC}_\bullet}$ & $13\%$ & $2969$ & $7.5 \times 10^{-6}$\\

 & NALU & $26\%$ & $3862$ & $9.2 \times 10^{-6}$\\

\multirow{-3}{*}{\raggedleft\arraybackslash mul} & NMU & $94\%$ & $1677$ & $3 \times 10^{-6}$\\
\bottomrule
\end{tabular}
\end{table}
