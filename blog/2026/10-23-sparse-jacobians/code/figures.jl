## The figure of the post, saved in its figures folder, for the light and the dark page.
using CairoMakie
figdir = mkpath(joinpath(@__DIR__, "..", "figures"))
# The look of the figures of the FEM post: bold Computer Modern, the font of the math in
# the post, and a horizontal legend above the plot. A unit is saved as half a point, so the
# figure has the 310 pt width of an example in the PDF and text at the 8 pt of the captions.
cm = Makie.texfont(:bold)
look = Theme(fonts=(; regular=cm, bold=cm), fontsize=16, size=(620, 260), backgroundcolor=:transparent,
             Lines=(; linewidth=2.5), Scatter=(; markersize=10),
             Legend=(; orientation=:horizontal, framevisible=false, tellheight=true, tellwidth=false))
# Color of axes and text, and color of the grid, drawn opaque
colors(fg, grid) = Theme(textcolor=fg, Axis=(; backgroundcolor=:transparent, xgridcolor=grid, ygridcolor=grid,
                         xtickcolor=fg, ytickcolor=fg, leftspinecolor=fg, rightspinecolor=fg,
                         topspinecolor=fg, bottomspinecolor=fg))
themes = (("", :black, "#e0e0e0"), ("_dark", "#dddddd", "#3a3a3a"))

# The estimation of the heating parameter for a small problem: the temperatures for λ = 2,
# the noisy data, and the temperatures of the estimate
include("heating_lifted.jl")
n = 50
t = (1:n) ./ (n + 1)
data = measurements(n; σ=0.03) # more noise than the timings, to show in the figure
model = estimation_model(data; lifted=true)
optimize!(model)
λ̂ = value(model[:λ])
for (suffix, fg, grid) in themes
    with_theme(merge(colors(fg, grid), look)) do
        fig = Figure()
        ax = Axis(fig[2, 1], xlabel="Position", ylabel="Temperature")
        lines!(ax, t, measurements(n; σ=0.0), label="λ = 2")
        scatter!(ax, t, data, label="Data", color=Makie.wong_colors()[3])
        lines!(ax, t, value.(model[:x]), label="Estimate, λ = $(round(λ̂; digits=3))", linestyle=:dash)
        Legend(fig[1, 1], ax)
        save(joinpath(figdir, "heating_estimate$suffix.pdf"), fig; pt_per_unit=0.5)
    end
end
