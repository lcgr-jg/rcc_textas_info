# ── Crude intake time series by refinery (dropdown) ───────────────────────────
def refinery_crude_kbd_dropdown(df, coverage, chartable_ids):
    """Single-series kbd chart with one trace per refinery; default = highest avg kbd."""
    plot_df = df[df['id'].isin(chartable_ids) & (df['kbd'] > 0)].copy()
    if plot_df.empty:
        print('No refineries with positive crude intake to chart.')
        return go.Figure()

    ref_stats = (
        plot_df.groupby(['id', 'plant_name', 'district'], as_index=False)
        .agg(avg_kbd=('kbd', 'mean'))
        .sort_values('avg_kbd', ascending=False)
    )

    fig = go.Figure()
    labels = []
    for _, row in ref_stats.iterrows():
        sub = plot_df[plot_df['id'] == row['id']].sort_values('period_dt')
        label = f"{row['plant_name'] or row['id']} | {row['district']}"
        labels.append(label)
        fig.add_trace(go.Scatter(
            x=sub['period_dt'],
            y=sub['kbd'],
            mode='lines',
            name='kbd',
            line=dict(color='#2a6f7a', width=2),
            visible=False,
        ))

    fig.data[0].visible = True
    default_label = labels[0]

    buttons = []
    for i, label in enumerate(labels):
        visible = [j == i for j in range(len(labels))]
        buttons.append(dict(
            label=label,
            method='update',
            args=[
                {'visible': visible},
                {'title': f'Crude intake — {label}'},
            ],
        ))

    fig.update_layout(
        title=f'Crude intake — {default_label}',
        xaxis=dict(
            tickformat='%d/%m/%Y',
            dtick='M3',
            tickangle=-90,
            showgrid=False,
        ),
        yaxis=dict(title='kbd', gridcolor='#e0e0e0', zeroline=False),
        hovermode='x unified',
        height=480,
        plot_bgcolor='white',
        margin=dict(t=90, b=120),
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.0,
            'y': 1.18,
            'xanchor': 'left',
            'yanchor': 'top',
        }],
    )
    return fig


fig_crude_kbd = refinery_crude_kbd_dropdown(rrc_crude_by_refinery, refinery_coverage, chartable_ids)
fig_crude_kbd.show()