import torch
import matplotlib.pyplot as plt


def show_err_img(img1, img2, titles, suptitle, lower=False, save=False):
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img1.detach().cpu() ** .4545)
    ax[1].imshow(img2.detach().cpu() ** .4545)
    ax[2].imshow(torch.abs((img1.cpu() - img2.cpu())).mean(dim=-1).detach())
    for j in range(3):
        ax[j].axis('off')
        ax[j].set_title(titles[j])
        if lower:
            ax[j].invert_yaxis()
    plt.tight_layout()
    plt.suptitle(suptitle)

    if save is False:
        plt.show()


def show_images(init_img, ref_img, titles=None, suptitle=None, save=False, savepath=None):
    if titles is None:
        titles = ['Init', 'Reference', 'MAE']
    elif len(titles) == 2:
        titles.append('MAE')
    if suptitle is None: suptitle = ''

    if init_img.shape[1] == 1:
        init_img = torch.cat([init_img]*3, dim=1)
    if ref_img.shape[1] == 1:
        ref_img = torch.cat([ref_img]*3, dim=1)

    if init_img.squeeze().shape[0] == 3:
        init_img = init_img.squeeze().permute(1, 2, 0)
    if ref_img.squeeze().shape[0] == 3:
        ref_img = ref_img.squeeze().permute(1, 2, 0)

    show_err_img(init_img, ref_img, titles, suptitle, save=save)

    if save:
        plt.savefig(savepath)
        plt.close('all')
    else:
        plt.show()
