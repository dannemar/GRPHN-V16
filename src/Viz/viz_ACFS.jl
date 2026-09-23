using GLMakie
using Statistics
using Printf

function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    num_records = div(length(raw_data), R)

    mat = reshape(raw_data, (R, num_records))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, num_records
end

function plot_all_observables()
    data_dir = "../../data/"

    # Define the list of configurations: (W, H, Target Energy)
    lattice_configs = [(128, 128, 100), (256, 256, 100),(512, 512, 100) ]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 100
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)

    eq_step = floor(Int, eqt / dt)
    eq_step_adj = div(eq_step, of) * of
    eq_idx_c = div(eq_step_adj, of)
    eq_idx = eq_idx_c + 1

    # Plot reduced to 2 subplots
    fig = Figure(size=(1600, 1000), fontsize=23)
    lwidth = 5.0

    ax_auto_ce = Axis(fig[1, 1], xlabel="Time (Post-Equilibration)", ylabel="Energy SCF C_E(t)", yscale = log10)
    ax_auto_cv = Axis(fig[2, 1], xlabel="Time (Post-Equilibration)", ylabel="Velocity SCF C_V(t)", yscale = log10)

    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"

        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        # Note spatial function files map to scf_ prefix instead of acf_
        mean_ce, _, _, num_records = process_observable(joinpath(data_dir, "scf_C_E" * strEnd * ".bin"), R)
        mean_cv, _, _, _ = process_observable(joinpath(data_dir, "scf_C_v" * strEnd * ".bin"), R)

        t_phy = (0:num_records-1) .* (dt * of)
        t_plot = t_phy[eq_idx:end] .- t_phy[eq_idx]

        lines!(ax_auto_ce, t_plot, abs.(mean_ce[eq_idx:end]), color=c, linewidth=lwidth, label="C_E ($lbl)")
        lines!(ax_auto_cv, t_plot, abs.(mean_cv[eq_idx:end]), color=c, linewidth=lwidth, label="C_V ($lbl)")
    end

    axislegend(ax_auto_ce, position=:rt)
    axislegend(ax_auto_cv, position=:rt)

    save("../../img/SCFs_LINLIN_combined_MULT.png", fig)
end

plot_all_observables()
