# This file was generated, do not modify it. # hide
#hideall
title!(diag1, "Gredy"); title!(rdiag1, "Randomized"); ylims!(rdiag1, ylims(diag1));
plot(diag1,rdiag1, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank1_diags.png")) # hide
title!(diag2, "Gredy"); title!(rdiag2, "Randomized"); ylims!(rdiag2, ylims(diag2));
plot(diag2,rdiag2, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank2_diags.png")) # hide
title!(diag3, "Gredy"); title!(rdiag3, "Randomized"); ylims!(rdiag3, ylims(diag3));
plot(diag3,rdiag3, layout=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "rank3_diags.png")) # hide
title!(diag4, "Gredy"); title!(rdiag4, "Randomized"); ylims!(rdiag4, ylims(diag4));
plot(diag4,rdiag4,dpi=300)
savefig(joinpath(@OUTPUT, "rank4_diags.png")) # hide