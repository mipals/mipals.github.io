# This file was generated, do not modify it. # hide
#hideall
Gka3_res(x,y,i,j,k)   = Gka2_res(x,y,i,j) - Gka2_res(X[k],y,i,j)*Gka2_res(x,X[k],i,j)/Gka2_res(X[k],X[k],i,j)
Gka4_app(x,y,i,j,k,m) = Gka3_res(y,X[m],i,j,k)/Gka3_res(X[m],X[m],i,j,k) * Gka3_res(x,X[m],i,j,k)
Gka4(x,y,i,j,k,m) = Gka3(x,y,i,j,k) + Gka4_app(x,y,i,j,k,m)
#hideall
i,j,k,m = sg[1], sg[2], sg[3], sg[4]
xs = X[[i;j;k;m]]
app_plot = contour(Xc,Xc, (x,y) ->  Gka4(x,y,i,j,k,m), fill=true,aspect_ratio=:equal, clim=(0,1))
scatter!(app_plot,xs,xs,legend=false,color=:pink,dpi=300)
title!("Rank-4-approx"); xlabel!("x"); ylabel!("y"); yflip!(true)
savefig(joinpath(@OUTPUT, "rank4_approx.png")) # hide
res_plot = contour(Xc,Xc, (x,y) -> abs.(G(x,y) - Gka4(x,y,i,j,k,m)), fill=true,aspect_ratio=:equal)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sg[1:4])], X[Not(sg[1:4])], label=false, color=:red)
scatter!(res_plot, [X[sg[1:4]]], [X[sg[1:4]]], label=false, color=:pink)
scatter!(res_plot, [X[sg[5]]], [X[sg[5]]], label=false, color=:green)
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
diag4 = plot(Xc, (x) -> G(x,x) - Gka4(x,x,i,j,k,m),label=false,color=:blue); title!("Diagonal Values");
scatter!(diag4,X[Not(sg[1:4])], (x) -> G(x,x) - Gka4(x,x,i,j,k,m),label="Remaining Data", color=:red)
scatter!(diag4,X[sg[1:4]], (x) -> G(x,x) - Gka4(x,x,i,j,k,m),label="Chosen pivots", color=:pink)
scatter!(diag4,[X[sg[5]]], (x) -> G(x,x) - Gka4(x,x,i,j,k,m),label="Next pivot", color=:green)
plot(res_plot,diag4, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank4_diag.png")) # hide
#hideall
i,j,k,m = sr[1], sr[2], sr[3], sr[4]
xs = X[[i;j;k;m]]
app_plot = contour(Xc,Xc, (x,y) ->  Gka4(x,y,i,j,k,m), fill=true,aspect_ratio=:equal, clim=(0,1))
scatter!(app_plot,xs,xs,legend=false,dpi=300)
title!("Rank-4-approx"); xlabel!("x"); ylabel!("y"); yflip!(true)
savefig(joinpath(@OUTPUT, "rank4_approx_random.png")) # hide
res_plot = contour(Xc,Xc, (x,y) -> abs.(G(x,y) - Gka4(x,y,i,j,k,m)), fill=true,aspect_ratio=:equal)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sr[1:4])], X[Not(sr[1:4])], label=false, color=:red)
scatter!(res_plot, [X[sr[1:4]]], [X[sr[1:4]]], label=false, color=:pink)
scatter!(res_plot, [X[sr[5]]], [X[sr[5]]], label=false, color=:green)
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
rdiag4 = plot(Xc, (x) -> G(x,x) - Gka4(x,x,i,j,k,m),label=false,color=:blue); title!("Diagonal Values");
scatter!(rdiag4,X[Not(sr[1:4])], (x) -> G(x,x) - Gka4(x,x,i,j,k,m),label="Remaining Data", color=:red)
scatter!(rdiag4,X[sr[1:4]], (x) -> G(x,x) - Gka4(x,x,i,j,k,m),label="Chosen pivots", color=:pink)
scatter!(rdiag4,[X[sr[5]]], (x) -> G(x,x) - Gka4(x,x,i,j,k,m),label="Next pivot", color=:green)
plot(res_plot,rdiag4, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank4_diag_random.png")) # hide