# -*- coding: utf-8 -*-

from antispoofing.sfsnet.datasets.replayattack import ReplayAttack
from antispoofing.sfsnet.datasets.casia import Casia
from antispoofing.sfsnet.datasets.oulunpu import OuluNPU
from antispoofing.sfsnet.datasets.casiara import CasiaRa
from antispoofing.sfsnet.datasets.casiareplayattack import CasiaReplayAttack
from antispoofing.sfsnet.datasets.uvad import UVAD
from antispoofing.sfsnet.datasets.casiarauvad import CasiaRaUVAD


registered_datasets = {0: ReplayAttack,
                       1: Casia,
                       2: OuluNPU,
                       4: CasiaRa,
                       5: UVAD,
                       6: CasiaReplayAttack,
                       7: CasiaRaUVAD,
                       }
