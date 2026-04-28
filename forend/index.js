const cav = document.getElementById('cav')
const ctx = cav.getContext('2d')

const numTocolor = {
    1: "rgb(255, 10, 10)",
    0: "rgb(1, 1, 1)",
    3: "rgb(10, 255, 1)",
    4: "rgb(255, 255, 255)",
}

const X = 28
const Y = 28

const Img = new Array(Y).fill(0).map(() => new Array(X).fill(numTocolor[0]))

const fill = (num) => {
    for (let y = 0; y < Y; y++) {
        for (let x = 0; x < X; x++) {
            Img[y][x] = numTocolor[num]
        }
    }
}

const render = () => {
    for (let x = 0; x < X; x++) {
        for (let y = 0; y < Y; y++) {
            ctx.fillStyle = Img[y][x]
            ctx.fillRect(x * cav.width / X, y * cav.height / Y, cav.width / X, cav.height / Y)
        }
    }
}
render()

let d = false
document.getElementById("ok").style.visibility = "hidden"

cav.addEventListener("mousedown", () => {
    d = true
    document.getElementById("ok").style.visibility = "hidden"
})
cav.addEventListener("mouseup", () => {
    d = false
    document.getElementById("ok").style.visibility = "visible"
})

const fo = document.querySelectorAll("input")
cav.addEventListener("mousemove", (e) => {
    if (!d) return
    let color = 1
    fo.forEach((it) => {
        if (it.checked === true) {
            color = it.value
        }
    })
    let y = Math.floor(e.offsetY / cav.height * Y)
    let x = Math.floor(e.offsetX / cav.width * X)
    if (y >= 0 && y < Y && x >= 0 && x < X) {
        Img[y][x] = numTocolor[color]
        render()
    }
}, false)

const reset = document.getElementById("reset")
reset.addEventListener("mousedown", () => {
    fill(0)
    render()
    document.getElementById("res").innerHTML = "<h2>PLEASE WRITE A NUMBER</h2>"
})

// ==================== 神经网络相关代码 ====================

const LAYER0_SIZE = 400
const LAYER1_SIZE = 200
const LAYER2_SIZE = 60
const LAYER3_SIZE = 100
const LAYER4_SIZE = 10
const LAYER_NUM = 5

let nodeValues = []

let networkData = null

const z = (x) => {
    return x > 0 ? x : 0.01 * x
}

const e = (x) => {
    return Math.exp(x)
}

const loadModel = async () => {
    try {
        const response = await fetch('network-2.json')
        networkData = await response.json()
        console.log("模型加载成功喵～")
        console.log("网络结构：", networkData.layers.length, "层")
        networkData.layers.forEach((layer, idx) => {
            console.log(`层 ${idx}: ${layer.length} 个节点`)
        })
    } catch (err) {
        console.error("加载模型失败喵：", err)
    }
}

const initVal = (input) => {
    nodeValues = []
    nodeValues.push([...input])
    for (let l = 1; l < LAYER_NUM; l++) {
        let size = 0
        switch(l) {
            case 1: size = LAYER1_SIZE; break
            case 2: size = LAYER2_SIZE; break
            case 3: size = LAYER3_SIZE; break
            case 4: size = LAYER4_SIZE; break
        }
        nodeValues.push(new Array(size).fill(0))
    }
}

const cleanHidingAndOutputLayers = () => {
    for (let l = 1; l < LAYER_NUM; l++) {
        nodeValues[l].fill(0)
    }
}

const forwardSpread = () => {
    cleanHidingAndOutputLayers()

    for (let l = 0; l < LAYER_NUM - 1; l++) {
        const layerData = networkData.layers[l]
        
        for (let i = 0; i < layerData.length; i++) {
            const node = layerData[i]
            let activatedVal = 0

            if (l === 0) {
                activatedVal = nodeValues[l][i]
            } else {
                activatedVal = z(nodeValues[l][i] - node.b)
            }

            for (let j = 0; j < node.w.length; j++) {
                nodeValues[l + 1][j] += activatedVal * node.w[j]
            }
        }
    }
}

const getFinalNodesVal = () => {
    const finalNodesVal = new Array(LAYER4_SIZE)
    let max = -100, min = 100
    
    for (let i = 0; i < LAYER4_SIZE; i++) {
        const val = nodeValues[LAYER_NUM - 1][i]
        if (val > max) max = val
        if (val < min) min = val
    }
    
    for (let i = 0; i < LAYER4_SIZE; i++) {
        const val = nodeValues[LAYER_NUM - 1][i]
        finalNodesVal[i] = e(val - max)
    }
    
    return finalNodesVal
}

const predict = (imgData) => {
    if (!networkData || !networkData.layers) {
        console.error("模型还没加载喵！")
        return -1
    }

    const poolified = new Array(20 * 20).fill(0)
    const df = 28 / 20, ds = 28 / 20

    for (let f = 0; f < 20; f++) {
        for (let s = 0; s < 20; s++) {
            const begin_f = df * f
            const end_f = df * (f + 1)
            const begin_s = ds * s
            const end_s = ds * (s + 1)
            let maxWeightedV = 0.0

            for (let f_i = Math.floor(begin_f); f_i < end_f; f_i++) {
                for (let s_i = Math.floor(begin_s); s_i < end_s; s_i++) {
                    const pixelBeginF = f_i
                    const pixelEndF = f_i + 1
                    const pixelBeginS = s_i
                    const pixelEndS = s_i + 1

                    const overlapF = Math.min(end_f, pixelEndF) - Math.max(begin_f, pixelBeginF)
                    const overlapS = Math.min(end_s, pixelEndS) - Math.max(begin_s, pixelBeginS)
                    const overlapArea = overlapF * overlapS

                    let pixelVal = 0
                    if (f_i >= 0 && f_i < 28 && s_i >= 0 && s_i < 28) {
                        const color = imgData[f_i][s_i]
                        if (color === numTocolor[1] || color === numTocolor[3] || color === numTocolor[4]) {
                            pixelVal = 1.0
                        }
                    }

                    const weightedVal = pixelVal * overlapArea
                    if (maxWeightedV < weightedVal) {
                        maxWeightedV = weightedVal
                    }
                }
            }
            poolified[f * 20 + s] = maxWeightedV
        }
    }

    initVal(poolified)
    forwardSpread()

    const results = getFinalNodesVal()

    let predictedClass = 0
    let maxValue = results[0]
    for (let j = 1; j < LAYER4_SIZE; j++) {
        if (results[j] > maxValue) {
            maxValue = results[j]
            predictedClass = j
        }
    }

    return predictedClass
}

document.getElementById("ok").addEventListener("click", () => {
    const res = predict(Img)
    if (res >= 0) {
        document.getElementById("res").innerHTML = `<h2>Predict: ${res}</h2>`
    } else {
        document.getElementById("res").innerHTML = "<h2>ERROR: 模型加载失败喵</h2>"
    }
})

loadModel()
