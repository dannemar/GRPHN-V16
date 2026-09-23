using GLMakie
using Statistics
using Printf

function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    T_steps = div(length(raw_data), R)
    mat = reshape(raw_data, (R, T_steps))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, T_steps
end

function interactive_acf_fitter()
    data_dir = "../../data/"
    lattice_configs = [(100, 100, 500)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 25
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)
    eq_idx = Int(ceil(eqt / dt)) + 1

    # Main Figure Layout
    fig = Figure(size=(1800, 1200), fontsize=20)

    # Left Grid for Plots
    plot_grid = fig[1, 1] = GridLayout()

    # Right Grid for UI Controls
    ui_grid = fig[1, 2] = GridLayout(width=400)

    # Initialize log-linear axes
    ax_ce = Axis(plot_grid[1, 1], xlabel="Lag Time τ", ylabel="Energy C_E(τ)", yscale=log10)
    ax_cv = Axis(plot_grid[2, 1], xlabel="Lag Time τ", ylabel="Velocity C_V(τ)", yscale=log10)
    ax_crel = Axis(plot_grid[3, 1], xlabel="Lag Time τ", ylabel="Length C_rel(τ)", yscale=log10)

    # Create UI Sliders
    sg = SliderGrid(ui_grid[1, 1],
        (label = "A1", range = 0.0:0.01:1.0, startvalue = 0.5),
        (label = "τ1 (Fast)", range = 0.01:0.1:100.0, startvalue = 1.0),
        (label = "A2", range = 0.0:0.01:1.0, startvalue = 0.3),
        (label = "τ2 (Mid)", range = 1.0:1.0:500.0, startvalue = 50.0),
        (label = "A3", range = 0.0:0.01:1.0, startvalue = 0.2),
        (label = "τ3 (Slow)", range = 10.0:10.0:5000.0, startvalue = 500.0)
    )

    A1 = sg.sliders[1].value
    t1 = sg.sliders[2].value
    A2 = sg.sliders[3].value
    t2 = sg.sliders[4].value
    A3 = sg.sliders[5].value
    t3 = sg.sliders[6].value

    colors = Makie.wong_colors()
    lwidth = 2.0

    # Observable array to hold the lag time for the exponentials
    t_lag_obs = Observable(Float64[])

    # Lift the exponential functions based on slider states
    exp1_func = @lift($A1 .* exp.(-$t_lag_obs ./ $t1))
    exp2_func = @lift($A2 .* exp.(-$t_lag_obs ./ $t2))
    exp3_func = @lift($A3 .* exp.(-$t_lag_obs ./ $t3))
    exp_sum = @lift($exp1_func .+ $exp2_func .+ $exp3_func)

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"
        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of)"

        # Process data (Excluding C_D)[cite: 5]
        mean_ce, _, _, T_steps = process_observable(joinpath(data_dir, "acf_C_E" * strEnd * ".bin"), R)
        mean_cv, _, _, _ = process_observable(joinpath(data_dir, "acf_C_v" * strEnd * ".bin"), R)
        mean_crel, _, _, _ = process_observable(joinpath(data_dir, "acf_C_rel" * strEnd * ".bin"), R)

        # Calculate lag time starting at 0 for the fitted decays
        len_decay = length(mean_ce[eq_idx:end])
        t_lag = (0:(len_decay-1)) .* dt

        # Update observable for the lifted functions
        t_lag_obs[] = t_lag

        # Plot Raw Data (Negative values will be ignored by Makie's log10 scale)
        lines!(ax_ce, t_lag, mean_ce[eq_idx:end], color=(c, 0.4), linewidth=lwidth+1, label="Raw C_E")
        lines!(ax_cv, t_lag, mean_cv[eq_idx:end], color=(c, 0.4), linewidth=lwidth+1, label="Raw C_V")
        lines!(ax_crel, t_lag, mean_crel[eq_idx:end], color=(c, 0.4), linewidth=lwidth+1, label="Raw C_rel")
    end

    # Overlay Interactive Exponentials on all axes
    for ax in [ax_ce, ax_cv, ax_crel]
        lines!(ax, t_lag_obs, exp1_func, color=:red, linewidth=lwidth, linestyle=:dash, label="Exp 1")
        lines!(ax, t_lag_obs, exp2_func, color=:blue, linewidth=lwidth, linestyle=:dash, label="Exp 2")
        lines!(ax, t_lag_obs, exp3_func, color=:green, linewidth=lwidth, linestyle=:dash, label="Exp 3")
        lines!(ax, t_lag_obs, exp_sum, color=:black, linewidth=lwidth+0.5, label="Sum")
    end

    axislegend(ax_ce, position=:rt)
    axislegend(ax_cv, position=:rt)
    axislegend(ax_crel, position=:rt)

    display(fig)
end

interactive_acf_fitter()
