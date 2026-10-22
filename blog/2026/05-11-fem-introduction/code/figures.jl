## Result figures of the post, saved in its figures folder, for the light and the dark page.
using CairoMakie
figdir = mkpath(joinpath(@__DIR__, "..", "figures"))
# The look of every figure: bold Computer Modern, the font of the math in the post, and a
# horizontal legend above the plots. A unit of the figures is saved as half a point, so they
# have the 310 pt width of an example in the PDF and text at the 8 pt of the captions, a
# little smaller than the 9.4 pt of the examples. Makie's default palette is the colorblind-safe one of Wong.
cm = Makie.texfont(:bold)
look = Theme(fonts=(; regular=cm, bold=cm), fontsize=16, size=(620,260), backgroundcolor=:transparent,
             Lines=(; linewidth=2.5), Scatter=(; markersize=12),
             Legend=(; orientation=:horizontal, framevisible=false, tellheight=true, tellwidth=false))
# Color of axes and text, and color of the grid. The grid is drawn opaque, as some PDF
# viewers draw transparent grid lines fully opaque.
colors(fg, grid) = Theme(textcolor=fg, Axis=(; backgroundcolor=:transparent, xgridcolor=grid, ygridcolor=grid,
                         xtickcolor=fg, ytickcolor=fg, leftspinecolor=fg, rightspinecolor=fg,
                         topspinecolor=fg, bottomspinecolor=fg))
themes = (("", :black, "#e0e0e0"), ("_dark", "#dddddd", "#3a3a3a"))

include("impedance_tube.jl")
for (suffix, fg, grid) in themes
    with_theme(merge(colors(fg, grid), look)) do
        fig = Figure()
        for (col, part, name) in ((1, real, "Real part"), (2, imag, "Imaginary part"))
            ax = Axis(fig[2,col], xlabel="Frequency (Hz)", ylabel=name)
            lines!(ax, freq, part.(Z), label="Analytical")
            lines!(ax, freq, part.(Z_num), label="FEM", linestyle=:dash)
        end
        Legend(fig[1,:], content(fig[2,1]))
        save(joinpath(figdir, "impedance_results$suffix.pdf"), fig; pt_per_unit=0.5)
    end
end

include("modes_fem.jl")
nmodes = 9
for (suffix, fg, grid) in themes
    with_theme(merge(colors(fg, grid), look)) do
        fig = Figure()
        ax = Axis(fig[2,1], xlabel="Mode number", xticks=1:nmodes, ylabel="Frequency (Hz)", limits=(nothing, (0, nothing)))
        scatter!(ax, 1:nmodes, analytical_freqs[2:nmodes+1], label="Analytical")
        scatter!(ax, 1:nmodes, freq[2:nmodes+1], label="FEM", marker=:xcross)
        Legend(fig[1,1], ax)
        save(joinpath(figdir, "modes_results$suffix.pdf"), fig; pt_per_unit=0.5)
    end
end
