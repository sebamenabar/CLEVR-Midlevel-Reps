import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_midreps(img, tgt_midreps, pred_midreps, num_samples=32):
    num_samples = min(img.size(0), num_samples)
    tasks = list(tgt_midreps.keys())
    cols_per_task = 3
    fig, axes = plt.subplots(
        nrows=num_samples,
        ncols=1 + cols_per_task * len(tasks),
        figsize=(6 * (1 + cols_per_task * len(tasks)), 6 * num_samples),
    )

    img = img.permute(0, 2, 3, 1).detach().cpu()
    for ns in range(num_samples):
        img_ax = axes[ns, 0]
        img_ax.imshow(img[ns])
        if ns == 0:
            img_ax.set_title("Input image", fontsize=24)
        img_ax.axis("off")

    for i, task_name in enumerate(tasks):

        task_tgt_midreps = tgt_midreps[task_name][:num_samples]
        task_pred_midreps = pred_midreps[task_name][:num_samples]
        task_midreps_dist = (task_tgt_midreps - task_pred_midreps).abs()
        task_midreps_dist = task_midreps_dist.mean(1)  # .unsqueeze(1)

        vmax = max(task_tgt_midreps.max(), task_pred_midreps.max(), 1)
        vmin = min(task_tgt_midreps.min(), task_pred_midreps.min(), 0)

        # gt_img = make_grid(task_tgt_midreps, nrow=1, padding=0).cpu()
        gt_img = task_tgt_midreps.cpu()
        # pred_img = make_grid(task_pred_midreps, nrow=1, padding=0).cpu()
        pred_img = task_pred_midreps.detach().cpu()
        # dist_img = make_grid(task_midreps_dist, nrow=1, padding=0).cpu()
        dist_img = task_midreps_dist.detach().cpu()

        # dist_img = dist_img[0]
        if task_tgt_midreps.size(1) == 1:
            colorbar = True
            gt_img = gt_img.squeeze(1)
            pred_img = pred_img.squeeze(1)
        elif task_tgt_midreps.size(1) == 3:
            colorbar = False
            gt_img = gt_img.permute(1, 2, 0)
            pred_img = pred_img.permute(1, 2, 0)

        if (0 > gt_img).any() or (gt_img > 1).any():
            print(f"gt img {task_name} out of range")
        if (0 > pred_img).any() or (pred_img > 1).any():
            print(f"pred img {task_name} out of range")

        for ns in range(num_samples):
            gt_ax = axes[ns, i * cols_per_task + 1]
            pred_ax = axes[ns, i * cols_per_task + 2]
            dist_ax = axes[ns, i * cols_per_task + 3]

            gt_handle = gt_ax.imshow(gt_img[ns], cmap="viridis", vmax=vmax, vmin=vmin)
            pred_handle = pred_ax.imshow(
                pred_img[ns], cmap="viridis", vmax=vmax, vmin=vmin
            )
            if colorbar:
                divider = make_axes_locatable(gt_ax)
                cax0 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(gt_handle, cax=cax0, use_gridspec=True)

                divider = make_axes_locatable(pred_ax)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(pred_handle, cax=cax1, use_gridspec=True)

            dist_handle = dist_ax.imshow(dist_img[ns], cmap="viridis")

            divider = make_axes_locatable(dist_ax)
            cax2 = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(dist_handle, cax=cax2, use_gridspec=True)

            if ns == 0:
                gt_ax.set_title(f"Real {task_name}", fontsize=24)
                pred_ax.set_title(f"Predicted {task_name}", fontsize=24)
                dist_ax.set_title(f"Abs Distance {task_name}", fontsize=24)
            gt_ax.axis("off")
            pred_ax.axis("off")
            dist_ax.axis("off")

    plt.subplots_adjust(wspace=0.08, hspace=0.05)
    # plt.axis("off")
    # self.log_figure(fig, f"{prefix}_validation", self.global_step)
    return fig