# This file was generated, do not modify it. # hide
#hideall
function PivotedCholesky(X,G,r; strategy=:rand)
    d = [G(x,x) for x in eachrow(X)]
    tr_true = sum(d)
    n = length(d)
    F = zeros(n,0)
    s = zeros(Int,0)
    if strategy==:uniform
        samples =  sample(1:n,r,replace=false)
    end
    for i = 1:r
        if strategy == :rand
            idx = sample(1:length(d), Weights(d/sum(d)), 1)[1]
        elseif strategy == :greedy
            idx = argmax(d)
        elseif strategy==:uniform
            idx = samples[i]
        end
        g = [G(X[idx,:], x) for x in eachrow(X)]
        g = g - F*F[idx,:]
        F = [F g / sqrt(abs(g[idx] + eps()))]
        d = d - abs.(F[:,i]).^2 # Update diagonal of residual
        d = max.(d,0)           # Ensure non-negative probability
        s = [s; idx]
    end
    err = tr_true - sum(dot(f,f) for f in eachrow(F)) # This is tr(K - F*F') 
    return F,s,err
end
r = 5
Fr,sr,errr = PivotedCholesky(X,G,r)
Fg,sg,errg = PivotedCholesky(X,G,r;strategy=:greedy)
Fu,su,erru = PivotedCholesky(X,G,r;strategy=:uniform)
#hideall
res_plot = contour(Xc,Xc, (x,y) -> G(x,y), fill=true,aspect_ratio=:equal, clim=(0,1))
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sg[1])], X[Not(sg[1])], label=false, color=:red)
scatter!(res_plot, [X[sg[1]]], [X[sg[1]]], label=false, color=:green)
diag0 = plot(Xc, (x) -> G(x,x),label=false,color=:blue); title!("Diagonal Values");
scatter!(diag0,X, (x) -> G(x,x),label="Remaining Data",color=:red); ylims!((0,1.1));
scatter!(diag0,[X[sg[1]]], (x) -> G(x,x),label="Next pivot",color=:green)
plot(res_plot,diag0,dpi=300)
savefig(joinpath(@OUTPUT, "rank0_diag.png")) # hide
#hideall
res_plot = contour(Xc,Xc, (x,y) -> G(x,y), fill=true,aspect_ratio=:equal, clim=(0,1))
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sg[1])], X[Not(sg[1])], label=false, color=:red)
scatter!(res_plot, [X[sg[1]]], [X[sg[1]]], label=false, color=:green)
diag0 = plot(Xc, (x) -> G(x,x),label=false,color=:blue); title!("Diagonal Values");
scatter!(diag0,X, (x) -> G(x,x),label="Remaining Data",color=:red); ylims!((0,1.1));
scatter!(diag0,[X[sr[1]]], (x) -> G(x,x),label="Next pivot",color=:green,dpi=300)
savefig(joinpath(@OUTPUT, "rank0_diag_random.png")) # hide
#hideall
Gka1(x,y,i=1)   = G(y,X[i])/G(X[i],X[i]) * G(x,X[i])
Gka1_res(x,y,i) = G(x,y) - G(X[i],y)*G(x,X[i])/G(X[i],X[i])
#hideall
idx = sg[1]
app_plot = contour(Xc,Xc, (x,y) ->  Gka1(x,y,idx), fill=true,aspect_ratio=:equal, clim=(0,1))
scatter!(app_plot,[X[idx]],[X[idx]],legend=false, color=:pink,dpi=300)
title!("Rank-1-approx"); xlabel!("x"); ylabel!("y"); yflip!(true)
savefig(joinpath(@OUTPUT, "rank1_approx.png")) # hide
res_plot = contour(Xc,Xc, (x,y) -> G(x,y) - Gka1(x,y,idx), fill=true,aspect_ratio=:equal, clim=(0,1))
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sg[1])], X[Not(sg[1])], label=false, color=:red)
scatter!(res_plot, [X[sg[1]]], [X[sg[1]]], label=false, color=:pink)
scatter!(res_plot, [X[sg[2]]], [X[sg[2]]], label=false, color=:green)
diag1 = plot(Xc, (x) -> G(x,x) - Gka1(x,x,idx),label=false,color=:blue); title!("Diagonal Values");
scatter!(diag1, X[Not(sg[1])], (x) -> G(x,x) - Gka1(x,x,idx),label="Remaining Data", color=:red)
scatter!(diag1, [X[sg[1]]], (x) -> G(x,x) - Gka1(x,x,idx),label="Chosen pivot", color=:pink)
scatter!(diag1, [X[sg[2]]], (x) -> G(x,x) - Gka1(x,x,idx),label="Next pivot",color=:green)
plot(res_plot, diag1, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank1_diag.png")) # hide
#hideall
idx = sr[1]
app_plot = contour(Xc,Xc, (x,y) ->  Gka1(x,y,idx), fill=true,aspect_ratio=:equal, clim=(0,1))
scatter!(app_plot,[X[idx]],[X[idx]],legend=false,color=:pink,dpi=300)
title!("Rank-1-approx"); xlabel!("x"); ylabel!("y"); yflip!(true)
savefig(joinpath(@OUTPUT, "rank1_approx_random.png")) # hide
res_plot = contour(Xc,Xc, (x,y) -> G(x,y) - Gka1(x,y,idx), fill=true,aspect_ratio=:equal, clim=(0,1))
title!(res_plot,"Residual"); xlabel!("x"); ylabel!("y"); yflip!(true)
plot!(res_plot,[-2; 2], [-2; 2], color=:blue, label=false);
scatter!(res_plot, X[Not(sr[1])], X[Not(sr[1])], label=false, color=:red)
scatter!(res_plot, [X[sr[1]]], [X[sr[1]]], label=false, color=:pink)
scatter!(res_plot, [X[sr[2]]], [X[sr[2]]], label=false, color=:green)
rdiag1 = plot(Xc, (x) -> G(x,x) - Gka1(x,x,idx),label=false,color=:blue); title!("Diagonal Values");
scatter!(rdiag1, X[Not(sr[1])], (x) -> G(x,x) - Gka1(x,x,idx),label="Remaining Data", color=:red)
scatter!(rdiag1, [X[sr[1]]], (x) -> G(x,x) - Gka1(x,x,idx),label="Chosen pivot", color=:pink)
scatter!(rdiag1, [X[sr[2]]], (x) -> G(x,x) - Gka1(x,x,idx),label="Next pivot", color=:green)
plot(res_plot,rdiag1, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank1_diag_random.png")) # hide