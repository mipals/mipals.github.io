## Result figure of the post, saved in its figures folder, for the light and the dark page.
using Plots
figdir = mkpath(joinpath(@__DIR__, "..", "figures"))
# Suffix, color of axes and text, and color of the grid. The grid is drawn opaque,
# as some PDF viewers draw transparent grid lines fully opaque.
themes = (("", :black, "#e0e0e0"), ("_dark", "#dddddd", "#3a3a3a"))
style(fg, grid) = (background_color=:transparent, foreground_color=fg, legend_background_color=:transparent,
             foreground_color_legend=fg, legend_font_color=fg, gridcolor=grid, gridalpha=1)

include("pulsating_sphere.jl")
for (suffix, fg, grid) in themes
    plot(ka, real(p_analytical)/(ρ*c); label="Real part, analytical", color=1, linewidth=2,
         xlabel="ka", ylabel="Normalized surface pressure", legend=:outertop, legend_column=2, size=(560,380),
         margin=5Plots.mm, style(fg, grid)...)
    plot!(ka, imag(p_analytical)/(ρ*c); label="Imaginary part, analytical", color=2, linewidth=2)
    scatter!(ka[1:2:end], real(p_bem[1:2:end])/(ρ*c); label="Real part, BEM", color=1, marker=:x, markersize=4, markerstrokewidth=1.5)
    scatter!(ka[1:2:end], imag(p_bem[1:2:end])/(ρ*c); label="Imaginary part, BEM", color=2, marker=:x, markersize=4, markerstrokewidth=1.5)
    savefig(joinpath(figdir, "pulsating_sphere_results$suffix.pdf"))
end
