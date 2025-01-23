# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np

EXPECTED_ANSWERS = [
    {  # test answers for PyTorch 2.0
        "integration_segmentation_3d": {
            "losses": [
                0.5430086106061935,
                0.47010003924369814,
                0.4453376233577728,
                0.451901963353157,
                0.4398456811904907,
                0.43450237810611725,
            ],
            "best_metric": 0.9329540133476257,
            "infer_metric": 0.9330471754074097,
            "output_sums": [
                0.14212507078546172,
                0.15199039602949577,
                0.15133471939291526,
                0.13967984811021827,
                0.18831614355832332,
                0.1694076821827231,
                0.14663931509271658,
                0.16788710637623733,
                0.1569452710008219,
                0.17907130698392254,
                0.16244092698688475,
                0.1679350345855819,
                0.14437674754879065,
                0.11355098478396568,
                0.161660275855964,
                0.20082478187698194,
                0.17575491677668853,
                0.0974593860605401,
                0.19366775441539907,
                0.20293016863409002,
                0.19610441127101647,
                0.20812173772459808,
                0.16184212006067655,
                0.13185211452732482,
                0.14824716961304257,
                0.14229818359602905,
                0.23141282114085215,
                0.1609268635938338,
                0.14825300029123678,
                0.10286266811772046,
                0.11873484714087054,
                0.1296615212510262,
                0.11386621034856693,
                0.15203351148564773,
                0.16300823766585265,
                0.1936726544485426,
                0.2227251185536394,
                0.18067789917505797,
                0.19005874127683337,
                0.07462121515702229,
            ],
        }
    },
    {  # test answers for PyTorch 1.12.1
        "integration_classification_2d": {
            "losses": [0.776835828070428, 0.1615355300011149, 0.07492854832938523, 0.04591309238865877],
            "best_metric": 0.9999184380485994,
            "infer_prop": [1029, 896, 980, 1033, 961, 1046],
        },
        "integration_segmentation_3d": {
            "losses": [
                0.5428894340991974,
                0.47331981360912323,
                0.4482289582490921,
                0.4452722787857056,
                0.4289989799261093,
                0.4359133839607239,
            ],
            "best_metric": 0.933259129524231,
            "infer_metric": 0.9332860708236694,
            "output_sums": [
                0.142167581604417,
                0.15195543400875847,
                0.1512754523215521,
                0.13962938779108452,
                0.18835719348918614,
                0.16943498693483486,
                0.1465709827477569,
                0.16806483607477135,
                0.1568844609697224,
                0.17911090857818554,
                0.16252098157181355,
                0.16806016936625395,
                0.14430124467305516,
                0.11316135548315168,
                0.16183771025615476,
                0.2009426314066978,
                0.1760258010156966,
                0.09700864497950844,
                0.1938495370314683,
                0.20319147575335647,
                0.19629641404249798,
                0.20852344793102826,
                0.16185073630020633,
                0.13184196857669161,
                0.1480959525354053,
                0.14232924377085415,
                0.23177739882790951,
                0.16094610375534632,
                0.14832771888168225,
                0.10259365443625812,
                0.11850632233099603,
                0.1294100326098242,
                0.11364228279017609,
                0.15181947897584674,
                0.16319358155815072,
                0.1940284526521386,
                0.22306137879066443,
                0.18083137638759522,
                0.1903135237574692,
                0.07402317520619131,
            ],
        },
        "integration_workflows": {
            "best_metric": 0.9219646483659745,
            "infer_metric": 0.921751058101654,
            "output_sums": [
                0.14183664321899414,
                0.1513957977294922,
                0.13804054260253906,
                0.13356828689575195,
                0.18456125259399414,
                0.16363763809204102,
                0.14090299606323242,
                0.16649389266967773,
                0.15651893615722656,
                0.17655134201049805,
                0.16116666793823242,
                0.1644763946533203,
                0.14383649826049805,
                0.11055326461791992,
                0.16080379486083984,
                0.19629907608032227,
                0.17441415786743164,
                0.053577423095703125,
                0.19043779373168945,
                0.19904804229736328,
                0.19526052474975586,
                0.20304107666015625,
                0.16030025482177734,
                0.13170623779296875,
                0.15118932723999023,
                0.13686418533325195,
                0.22668886184692383,
                0.1611471176147461,
                0.1472463607788086,
                0.10427379608154297,
                0.11962461471557617,
                0.1305704116821289,
                0.11204910278320312,
                0.15171337127685547,
                0.15962505340576172,
                0.18976259231567383,
                0.21649456024169922,
                0.17761802673339844,
                0.18516874313354492,
                0.03636503219604492,
            ],
            "best_metric_2": 0.9219559609889985,
            "infer_metric_2": 0.9217371672391892,
            "output_sums_2": [
                0.14187288284301758,
                0.15140819549560547,
                0.13802719116210938,
                0.1335887908935547,
                0.18454980850219727,
                0.1636652946472168,
                0.14091157913208008,
                0.16653108596801758,
                0.15651702880859375,
                0.17658615112304688,
                0.1611957550048828,
                0.16448307037353516,
                0.14385128021240234,
                0.1105203628540039,
                0.16085100173950195,
                0.19626951217651367,
                0.17442035675048828,
                0.053586483001708984,
                0.19042730331420898,
                0.1990523338317871,
                0.1952815055847168,
                0.20303773880004883,
                0.16034317016601562,
                0.13172531127929688,
                0.15118741989135742,
                0.1368694305419922,
                0.22667837142944336,
                0.16119050979614258,
                0.14726591110229492,
                0.10426473617553711,
                0.11961841583251953,
                0.13054800033569336,
                0.11203193664550781,
                0.15172529220581055,
                0.15963029861450195,
                0.18975019454956055,
                0.21646499633789062,
                0.17763566970825195,
                0.18517112731933594,
                0.03638744354248047,
            ],
        },
    },
    {  # test answers for cuda 10.x
        "integration_classification_2d": {
            "losses": [0.777176220515731, 0.16019743723664315, 0.07480076164197011, 0.045643698364780966],
            "best_metric": 0.9999418774120775,
            "infer_prop": [1030, 897, 980, 1033, 960, 1048],
        },
        "integration_segmentation_3d": {
            "losses": [
                0.5326887160539627,
                0.4685510128736496,
                0.46245276033878324,
                0.4411882758140564,
                0.4198471873998642,
                0.43021280467510226,
            ],
            "best_metric": 0.931993305683136,
            "infer_metric": 0.9326668977737427,
            "output_sums": [
                0.1418775228871769,
                0.15188869120317386,
                0.15140863737688195,
                0.1396146850007127,
                0.18784343811575696,
                0.16909487431163164,
                0.14649608249452073,
                0.1677767130878611,
                0.1568122289811143,
                0.17874181729735056,
                0.16213703658980205,
                0.16754335171970686,
                0.14444824920997243,
                0.11432402622850306,
                0.16143210936221247,
                0.20055289634107482,
                0.17543571757219317,
                0.09920729163334538,
                0.19297325815057875,
                0.2023200127892273,
                0.1956677579845722,
                0.20774045016425718,
                0.16193278944159428,
                0.13174198906539808,
                0.14830508550670007,
                0.14241105864278342,
                0.23090631643085724,
                0.16056153813499532,
                0.1480353269419819,
                0.10318719171632634,
                0.11867462580989198,
                0.12997011485830187,
                0.11401220332210203,
                0.15242746700662088,
                0.1628489107974574,
                0.19327235354175412,
                0.22184902863377548,
                0.18028049625972334,
                0.18958059106892552,
                0.07884601267057013,
            ],
        },
        "integration_workflows": {
            "best_metric": 0.9217087924480438,
            "infer_metric": 0.9214379042387009,
            "output_sums": [
                0.14209461212158203,
                0.15126705169677734,
                0.13800382614135742,
                0.1338181495666504,
                0.1850571632385254,
                0.16372442245483398,
                0.14059066772460938,
                0.16674423217773438,
                0.15653657913208008,
                0.17690563201904297,
                0.16154909133911133,
                0.16521310806274414,
                0.14388608932495117,
                0.1103353500366211,
                0.1609959602355957,
                0.1967010498046875,
                0.1746964454650879,
                0.05329275131225586,
                0.19098854064941406,
                0.19976520538330078,
                0.19576644897460938,
                0.20346736907958984,
                0.1601848602294922,
                0.1316051483154297,
                0.1511220932006836,
                0.13670969009399414,
                0.2276287078857422,
                0.1611800193786621,
                0.14751672744750977,
                0.10413789749145508,
                0.11944007873535156,
                0.1305546760559082,
                0.11204719543457031,
                0.15145111083984375,
                0.16007614135742188,
                0.1904129981994629,
                0.21741962432861328,
                0.17812013626098633,
                0.18587207794189453,
                0.03605222702026367,
            ],
            "best_metric_2": 0.9210659921169281,
            "infer_metric_2": 0.9208109736442566,
            "output_sums_2": [
                0.14227628707885742,
                0.1515035629272461,
                0.13819408416748047,
                0.13402271270751953,
                0.18525266647338867,
                0.16388607025146484,
                0.14076614379882812,
                0.16694307327270508,
                0.15677356719970703,
                0.1771831512451172,
                0.16172313690185547,
                0.1653728485107422,
                0.14413118362426758,
                0.11057281494140625,
                0.16121912002563477,
                0.19680166244506836,
                0.1748638153076172,
                0.053426265716552734,
                0.19117307662963867,
                0.19996356964111328,
                0.1959366798400879,
                0.20363712310791016,
                0.16037797927856445,
                0.13180780410766602,
                0.1513657569885254,
                0.13686084747314453,
                0.2277364730834961,
                0.16137409210205078,
                0.1476879119873047,
                0.10438394546508789,
                0.11967992782592773,
                0.13080739974975586,
                0.11226606369018555,
                0.15168476104736328,
                0.1602616310119629,
                0.190582275390625,
                0.21756458282470703,
                0.17825984954833984,
                0.18604803085327148,
                0.036206722259521484,
            ],
        },
    },
    {  # test answers for PyTorch 1.9
        "integration_workflows": {
            "output_sums_2": [
                0.14213180541992188,
                0.15153264999389648,
                0.13801145553588867,
                0.1338348388671875,
                0.18515968322753906,
                0.16404008865356445,
                0.14110612869262695,
                0.16686391830444336,
                0.15673542022705078,
                0.1772594451904297,
                0.16174745559692383,
                0.16518878936767578,
                0.1440296173095703,
                0.11033201217651367,
                0.1611781120300293,
                0.19660568237304688,
                0.17468547821044922,
                0.053053855895996094,
                0.1909656524658203,
                0.19952869415283203,
                0.1957845687866211,
                0.2034916877746582,
                0.16042661666870117,
                0.13193607330322266,
                0.15104389190673828,
                0.13695430755615234,
                0.22720861434936523,
                0.16157913208007812,
                0.14759159088134766,
                0.10379791259765625,
                0.11937189102172852,
                0.1306462287902832,
                0.11205482482910156,
                0.15182113647460938,
                0.16006708145141602,
                0.19011592864990234,
                0.21713829040527344,
                0.17794132232666016,
                0.18584394454956055,
                0.03577899932861328,
            ]
        },
        "integration_segmentation_3d": {  # for the mixed readers
            "losses": [
                0.5645154356956482,
                0.4984356611967087,
                0.472334086894989,
                0.47419720590114595,
                0.45881829261779783,
                0.43097741305828097,
            ],
            "best_metric": 0.9325698614120483,
            "infer_metric": 0.9326590299606323,
        },
    },
    {  # test answers for PyTorch 1.13
        "integration_workflows": {
            "output_sums_2": [
                0.14264830205979873,
                0.15264129328718357,
                0.1519652511118344,
                0.14003114557361543,
                0.18870416611118465,
                0.1699260498246968,
                0.14727475398203582,
                0.16870874483246967,
                0.15757932277023196,
                0.1797779694564011,
                0.16310501082450635,
                0.16850569170136015,
                0.14472958359864832,
                0.11402527744419455,
                0.16217657428257873,
                0.20135486560244975,
                0.17627557567092866,
                0.09802074024435596,
                0.19418729084978026,
                0.20339278025379662,
                0.1966174446916041,
                0.20872528599049203,
                0.16246183433492764,
                0.1323750751202327,
                0.14830347036335728,
                0.14300732028781024,
                0.23163101813922762,
                0.1612925258625139,
                0.1489573676973957,
                0.10299491921717041,
                0.11921404797064328,
                0.1300212751422368,
                0.11437829790254125,
                0.1524755276727056,
                0.16350584736767904,
                0.19424317961257148,
                0.2229762916892286,
                0.18121074825540173,
                0.19064286213535897,
                0.0747544243069024,
            ]
        },
        "integration_segmentation_3d": {  # for the mixed readers
            "losses": [
                0.5451162219047546,
                0.4709601759910583,
                0.45201429128646853,
                0.4443251401185989,
                0.4341257899999619,
                0.4350819975137711,
            ],
            "best_metric": 0.9316844940185547,
            "infer_metric": 0.9316383600234985,
        },
    },
    {  # test answers for cuda 12
        "integration_segmentation_3d": {
            "losses": [
                0.5362162500619888,
                0.4704935997724533,
                0.4335438072681427,
                0.4507470965385437,
                0.45187077224254607,
                0.4363303750753403,
            ],
            "best_metric": 0.9334161877632141,
            "infer_metric": 0.9335371851921082,
            "output_sums": [
                0.14210400101844414,
                0.1521489829835625,
                0.15127096315211278,
                0.13992817339153868,
                0.1884040828001848,
                0.16929503899789516,
                0.14662516818085808,
                0.16803982264111883,
                0.1570018930834878,
                0.17916684191571494,
                0.1626376090146162,
                0.1680113549677271,
                0.1446708736188978,
                0.1140289628362559,
                0.16191495673888556,
                0.20066696225510708,
                0.17581812459936835,
                0.09836918048666465,
                0.19355007524499268,
                0.20291004237066343,
                0.19606797329772976,
                0.2082113232291515,
                0.16189564397603906,
                0.13203990336741953,
                0.14849477534402156,
                0.14250633066863938,
                0.23139529505006795,
                0.16079877619802546,
                0.14821067071610583,
                0.10302449386782145,
                0.11876349315302756,
                0.13006925219380802,
                0.11431448379763984,
                0.15254606148569302,
                0.16317147221367873,
                0.19376668030880526,
                0.22260597124465822,
                0.18085088544070227,
                0.19010916899493174,
                0.07748195410499427,
            ],
        }
    },
    {  # test answers for 23.02
        "integration_segmentation_3d": {
            "losses": [
                0.5401686698198318,
                0.4789864182472229,
                0.4417317628860474,
                0.44183324575424193,
                0.4418945342302322,
                0.44213996827602386,
            ],
            "best_metric": 0.9316274523735046,
            "infer_metric": 0.9321609735488892,
            "output_sums": [
                0.14212507078546172,
                0.15199039602949577,
                0.15133471939291526,
                0.13967984811021827,
                0.18831614355832332,
                0.1694076821827231,
                0.14663931509271658,
                0.16788710637623733,
                0.1569452710008219,
                0.17907130698392254,
                0.16244092698688475,
                0.1679350345855819,
                0.14437674754879065,
                0.11355098478396568,
                0.161660275855964,
                0.20082478187698194,
                0.17575491677668853,
                0.0974593860605401,
                0.19366775441539907,
                0.20293016863409002,
                0.19610441127101647,
                0.20812173772459808,
                0.16184212006067655,
                0.13185211452732482,
                0.14824716961304257,
                0.14229818359602905,
                0.23141282114085215,
                0.1609268635938338,
                0.14825300029123678,
                0.10286266811772046,
                0.11873484714087054,
                0.1296615212510262,
                0.11386621034856693,
                0.15203351148564773,
                0.16300823766585265,
                0.1936726544485426,
                0.2227251185536394,
                0.18067789917505797,
                0.19005874127683337,
                0.07462121515702229,
            ],
        }
    },
    {  # test answers for 24.03
        "integration_segmentation_3d": {
            "losses": [
                0.5442982316017151,
                0.4741817444562912,
                0.4535954713821411,
                0.44163046181201937,
                0.4307525992393494,
                0.428487154841423,
            ],
            "best_metric": 0.9314384460449219,
            "infer_metric": 0.9315622448921204,
            "output_sums": [
                0.14268704426414708,
                0.1528672845845743,
                0.1521782248125706,
                0.14028769128068194,
                0.1889830671664784,
                0.16999075690664475,
                0.14736282992708227,
                0.16877952654821815,
                0.15779597155181269,
                0.17987829927082263,
                0.16320253928314676,
                0.16854299322173155,
                0.14497470986956967,
                0.11437140546369519,
                0.1624117412960871,
                0.20156009294443875,
                0.1764654154256958,
                0.0982348259217418,
                0.1942436068604293,
                0.20359421536407518,
                0.19661953116976483,
                0.2088326101468625,
                0.16273043545239807,
                0.1326107887439663,
                0.1489245275752285,
                0.143107476635514,
                0.23189027677929547,
                0.1613818424566088,
                0.14889532196775188,
                0.10332622984492143,
                0.11940054688302351,
                0.13040496302762658,
                0.11472123087193181,
                0.15307044007394474,
                0.16371989575844717,
                0.1942898223272055,
                0.2230120930471398,
                0.1814679187634795,
                0.19069496508164732,
                0.07537197031940022,
            ],
        }
    },
]


def test_integration_value(test_name, key, data, rtol=1e-2):
    for idx, expected in enumerate(EXPECTED_ANSWERS):
        if test_name not in expected:
            continue
        if key not in expected[test_name]:
            continue
        value = expected[test_name][key]
        if np.allclose(data, value, rtol=rtol):
            print(f"matched {idx} result of {test_name}, {key}, {rtol}.")
            return True
    raise ValueError(f"no matched results for {test_name}, {key}. {data}.")
