CONFIG = {
    'layers': [
        {
            'type': 'conv',
            'params':
                {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding': 1}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'dw_conv',
            'params':
                {'out_channels': 1024, 'kernel_size': 3, 'stride': 2, 'padding': 4}
        },
        {
            'type': 'conv',
            'params':
                {'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'padding': 0}
        },
        {
            'type': 'avg_pool',
            'params': {'kernel_size': 7, 'stride': 1}
        },
        {
            'type': 'flatten',
            'params': {}
        },
        {
            'type': 'fc',
            'params':
                {'in_features': 1024}
        },
    ]
}
