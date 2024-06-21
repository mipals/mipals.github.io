# This file was generated, do not modify it. # hide
#hideall
Gka2_app(x,y,i,j) = Gka1_res(y,X[j],i)/Gka1_res(X[j],X[j],i) * Gka1_res(x,X[j],i)
Gka2(x,y,i,j) = Gka1(x,y,i) + Gka2_app(x,y,i,j)
#hideall
i,j = sg[1], sg[2]
xs = X[[i;j]]
app_plot = contour(Xc,Xc, (x,y) ->  Gka2(x,y,i,j), fill=true,aspect_ratio=:equal, clim=(0,1))
scatter!(app_plot,xs,xs,legend=false,color=:pink,dpi=300)
title!("Rank-2-approx"); xlabel!("x"); ylabel!("y"); yflip!(true)
savefig(joinpath(@OUTPUT, "rank2_approx.png")) # hide
res_plot = contour(Xc,Xc, (x,y) -> G(x,y) - Gka2(x,y,i,j), fill=true,aspect_ratio=:equal)
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sg[1:2])], X[Not(sg[1:2])], label=false, color=:red)
scatter!(res_plot, [X[sg[1:2]]], [X[sg[1:2]]], label=false, color=:pink)
scatter!(res_plot, [X[sg[3]]], [X[sg[3]]], label=false, color=:green)
diag2 = plot(Xc, (x) -> G(x,x) - Gka2(x,x,i,j),label=false,color=:blue); title!("Diagonal Values");
scatter!(diag2,X[Not(sg[1:2])], (x) -> G(x,x) - Gka2(x,x,i,j),label="Remaining Data", color=:red)
scatter!(diag2,X[sg[1:2]], (x) -> G(x,x) - Gka2(x,x,i,j),label="Chosen pivots", color=:pink)
scatter!(diag2,[X[sg[3]]], (x) -> G(x,x) - Gka2(x,x,i,j),label="Next pivot", color=:green)
plot(res_plot,diag2, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank2_diag.png")) # hide
#hideall
i,j = sr[1], sr[2]
xs = X[[i;j]]
app_plot = contour(Xc,Xc, (x,y) ->  Gka2(x,y,i,j), fill=true,aspect_ratio=:equal, clim=(0,1))
scatter!(app_plot,xs,xs,legend=false,color=:pink)
title!("Rank-2-approx"); xlabel!("x"); ylabel!("y"); yflip!(true)
savefig(joinpath(@OUTPUT, "rank2_approx_random.svg")) # hide
res_plot = contour(Xc,Xc, (x,y) -> G(x,y) - Gka2(x,y,i,j), fill=true,aspect_ratio=:equal)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sr[1:2])], X[Not(sr[1:2])], label=false, color=:red)
scatter!(res_plot, [X[sr[1:2]]], [X[sr[1:2]]], label=false, color=:pink)
scatter!(res_plot, [X[sr[3]]], [X[sr[3]]], label=false, color=:green)
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
rdiag2 = plot(Xc, (x) -> G(x,x) - Gka2(x,x,i,j),label=false,color=:blue); title!("Diagonal Values");
scatter!(rdiag2, X[Not(sr[1:2])], (x) -> G(x,x) - Gka2(x,x,i,j),label="Remaining Data",color=:red)
scatter!(rdiag2, X[sr[1:2]], (x) -> G(x,x) - Gka2(x,x,i,j),label="Chosen pivots",color=:pink)
scatter!(rdiag2, [X[sr[3]]], (x) -> G(x,x) - Gka2(x,x,i,j),label="Next pivot",color=:green)
plot(res_plot,rdiag2, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank2_diag_random.png")) # hide