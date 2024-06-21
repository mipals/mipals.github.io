# This file was generated, do not modify it. # hide
#hideall
Gka2_res(x,y,i,j)   = Gka1_res(x,y,i) - Gka1_res(X[j],y,i)*Gka1_res(x,X[j],i)/Gka1_res(X[j],X[j],i)
Gka3_app(x,y,i,j,k) = Gka2_res(y,X[k],i,j)/Gka2_res(X[k],X[k],i,j) * Gka2_res(x,X[k],i,j)
Gka3(x,y,i,j,k) = Gka2(x,y,i,j) + Gka3_app(x,y,i,j,k)
#hideall
i,j,k = sg[1], sg[2], sg[3]
xs = [X[[i;j;k]]];
app_plot = contour(Xc,Xc, (x,y) ->  Gka3(x,y,i,j,k), fill=true,aspect_ratio=:equal, clim=(0,1))
scatter!(app_plot,xs,xs,legend=false,color=:pink,dpi=300)
title!("Rank-3-approx"); xlabel!("x"); ylabel!("y"); yflip!(true)
savefig(joinpath(@OUTPUT, "rank3_approx.png")) # hide
res_plot = contour(Xc,Xc, (x,y) -> abs.(G(x,y) - Gka3(x,y,i,j,k)), fill=true,aspect_ratio=:equal)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sg[1:3])], X[Not(sg[1:3])], label=false, color=:red)
scatter!(res_plot, [X[sg[1:3]]], [X[sg[1:3]]], label=false, color=:pink)
scatter!(res_plot, [X[sg[4]]], [X[sg[4]]], label=false, color=:green)
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
diag3 = plot(Xc, (x) -> G(x,x) - Gka3(x,x,i,j,k),label=false,color=:blue); title!("Diagonal Values");
scatter!(diag3, X[Not(sg[1:3])], (x) -> G(x,x) - Gka3(x,x,i,j,k),label="Remaining Data", color=:red)
scatter!(diag3, X[sg[1:3]], (x) -> G(x,x) - Gka3(x,x,i,j,k),label="Chosen pivots", color=:pink)
scatter!(diag3, [X[sg[4]]], (x) -> G(x,x) - Gka3(x,x,i,j,k),label="Next pivot", color=:green)
plot(res_plot,diag3, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank3_diag.png")) # hide
#hideall
i,j,k = sr[1], sr[2], sr[3]
xs = X[[i;j;k]];
app_plot = contour(Xc,Xc, (x,y) ->  Gka3(x,y,i,j,k), fill=true,aspect_ratio=:equal, clim=(0,1))
scatter!(app_plot,xs,xs,legend=false, color=:pink,dpi=300)
title!("Rank-3-approx"); xlabel!("x"); ylabel!("y"); yflip!(true)
savefig(joinpath(@OUTPUT, "rank3_approx_random.png")) # hide
res_plot = contour(Xc,Xc, (x,y) -> abs.(G(x,y) - Gka3(x,y,i,j,k)), fill=true,aspect_ratio=:equal)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sr[1:3])], X[Not(sr[1:3])], label=false, color=:red)
scatter!(res_plot, [X[sr[1:3]]], [X[sr[1:3]]], label=false, color=:pink)
scatter!(res_plot, [X[sr[4]]], [X[sr[4]]], label=false, color=:green)
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
rdiag3 = plot(Xc, (x) -> G(x,x) - Gka3(x,x,i,j,k),label=false,color=:blue); title!("Diagonal Values");
scatter!(rdiag3,X[Not(sr[1:3])], (x) -> G(x,x) - Gka3(x,x,i,j,k),label="Remaining Data", color=:red)
scatter!(rdiag3,X[sr[1:3]], (x) -> G(x,x) - Gka3(x,x,i,j,k),label="Chosen pivots", color=:pink)
scatter!(rdiag3,[X[sr[4]]], (x) -> G(x,x) - Gka3(x,x,i,j,k),label="Next pivot", color=:green)
plot(res_plot,rdiag3, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank3_diag_random.png")) # hide