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

function compute_temperature_acf(file_path::String, R::Int, eq_idx::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    num_records = div(length(raw_data), R)
    mat = reshape(raw_data, (R, num_records))

    # Slice the data to strictly evaluate post-equilibration steps
    T_post = mat[:, eq_idx:end]
    num_post_records = size(T_post, 2)

    # Temperature across realizations at the exact instance of equilibration
    T_t0 = T_post[:, 1]

    mean_T_t0 = mean(T_t0)
    mean_T_post = mean(T_post, dims=1)

    acf = zeros(Float64, num_post_records)
    for t in 1:num_post_records
        # Calculate expected value of the product minus the product of expected values
        acf[t] = mean(T_t0 .* T_post[:, t]) - (mean_T_t0 * mean_T_post[1, t])
    end

    # Normalize such that ACF(0) = 1
    if acf[1] != 0.0
        acf ./= acf[1]
    end

    return acf, num_records
end

function plot_all_observables()
    data_dir = "../../data/"

    # Define the list of configurations: (W, H, Target Energy)
    lattice_configs = [(256, 256, 100)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 20
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)

    eq_step = floor(Int, eqt / dt)
    eq_step_adj = div(eq_step, of) * of
    eq_idx_c = div(eq_step_adj, of)
    eq_idx = eq_idx_c + 1

    # Expanded figure layout to accommodate 3 subplots
    fig = Figure(size=(1600, 1500), fontsize=23)
    lwidth = 5.0

    ax_auto_ce = Axis(fig[1, 1],
        xlabel="Time (Post-Equilibration)",
        ylabel="Energy SCF C_E(t)",
        #yscale = log10,
    )

    ax_auto_cv = Axis(fig[2, 1],
        xlabel="Time (Post-Equilibration)",
        ylabel="Velocity SCF C_V(t)",
        #yscale = log10,
    )

    ax_auto_ct = Axis(fig[3, 1],
        xlabel="Time (Post-Equilibration)",
        ylabel="Temperature ACF C_T(t)",
        #yscale = log10,
    )

    colors = Makie.wong_colors()

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"

        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        mean_ce, _, _, num_records = process_observable(joinpath(data_dir, "scf_C_E" * strEnd * ".bin"), R)
        mean_cv, _, _, _ = process_observable(joinpath(data_dir, "scf_C_v" * strEnd * ".bin"), R)

        # Calculate the Temperature ACF from the standard temperature file
        temp_acf, _ = compute_temperature_acf(joinpath(data_dir, "temperature" * strEnd * ".bin"), R, eq_idx)

        t_phy = (0:num_records-1) .* (dt * of)
        t_plot = t_phy[eq_idx:end] .- t_phy[eq_idx]

        lines!(ax_auto_ce, t_plot, (mean_ce[eq_idx:end]), color=c, linewidth=lwidth, label="C_E ($lbl)")
        lines!(ax_auto_cv, t_plot, (mean_cv[eq_idx:end]), color=c, linewidth=lwidth, label="C_V ($lbl)")
        lines!(ax_auto_ct, t_plot, (temp_acf), color=c, linewidth=lwidth, label="C_T ($lbl)")
    end

    axislegend(ax_auto_ce, position=:rt)
    axislegend(ax_auto_cv, position=:rt)
    axislegend(ax_auto_ct, position=:rt)

    save("../../img/SCFs_ACFs_TEMP.png", fig)
end

plot_all_observables()
