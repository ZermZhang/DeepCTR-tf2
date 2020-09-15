#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : model_utils
@Software       : PyCharm
@Modify Time    : 2020/9/15 09:23     
@Author         : zermzhang
@version        : 1.0
@Desciption     : 
"""


def classes_activation_check(classes, activation):
    if classes == 2 and activation != 'sigmoid':
        raise Exception("{activation} not supported for {classes} classify".format(
            activation=activation, classes=classes
        ))
    if classes > 2 and activation == 'sigmoid':
        raise Exception("{actionvation} not supported for {classes} classify".format(
            actionvation=activation, classes=classes
        ))

    return 0