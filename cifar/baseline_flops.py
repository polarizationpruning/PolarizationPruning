from common import compute_conv_flops
from models import vgg16_linear
from models.resnet_expand import resnet56


def baseline_flops(num_classes, resnet_multi=0.7, vgg_multi=0.71):
    if resnet_multi is not None:
        model = resnet56(num_classes)
        res_baseline_flops = compute_conv_flops(model, cuda=True)
        print(f"Baseline FLOPs of CIFAR-{num_classes} ResNet-56: {res_baseline_flops:,}, 50% FLOPs: {res_baseline_flops / 2:,}")

        multi = resnet_multi
        model = resnet56(num_classes, width_multiplier=multi)
        flops = compute_conv_flops(model, cuda=True)
        print(f"FLOPs of CIFAR-{num_classes} ResNet-56 {multi}x: {flops:,}, FLOPs ratio: {flops / res_baseline_flops}")
        print()

    # from compute_flops import count_model_param_flops
    # flops_original_imple = count_model_param_flops(model, multiply_adds=False)
    # print(flops_original_imple)

    if vgg_multi is not None:
        model = vgg16_linear(num_classes)
        vgg_baseline_flops = compute_conv_flops(model)
        print(f"Baseline FLOPs of CIFAR-{num_classes} VGG-16: {vgg_baseline_flops:,}, 50% FLOPs: {vgg_baseline_flops / 2:,}")

        multi = vgg_multi
        model = vgg16_linear(num_classes, width_multiplier=multi)
        flops = compute_conv_flops(model)
        print(f"FLOPs of CIFAR-{num_classes} VGG-16 {multi}x: {flops:,}, FLOPs ratio: {flops / vgg_baseline_flops}")
        print()


def main():
    print("CIFAR-10 FLOPs computing:")
    # FLOPs of CIFAR-10 ResNet-56 0.32x: 14,225,096.0, FLOPs ratio: 0.11336029885031677
    baseline_flops(10, resnet_multi=0.32, vgg_multi=None)

    # FLOPs of CIFAR-10 ResNet-56 0.45x: 26,675,992.0, FLOPs ratio: 0.21258193443816895
    baseline_flops(10, resnet_multi=0.45, vgg_multi=None)

    # FLOPs of CIFAR-10 ResNet-56 0.56x: 36,924,264.0, FLOPs ratio: 0.2942507805829917
    baseline_flops(10, resnet_multi=0.56, vgg_multi=None)

    baseline_flops(10)

    print("CIFAR-100 FLOPs computing:")
    # FLOPs of CIFAR-100 ResNet-56 0.86x: 96,128,480.0, FLOPs ratio: 0.7660161341980126
    # FLOPs of CIFAR-100 VGG-16 0.78x: 188,885,145.0, FLOPs ratio: 0.6024854469661894
    baseline_flops(100, resnet_multi=0.86, vgg_multi=0.78)


if __name__ == '__main__':
    main()
