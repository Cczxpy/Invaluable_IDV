import math


class PriceMaker:
    def compute_all(total_count: int, price_ori: int, namelist, nameprice) -> str:
        decc = 0.97**total_count
        if decc <= 0.3:
            decc = 0.3
        decc = math.log10(10 * decc)
        if decc <= 0.65:
            decc = 0.65
        ans = price_ori * decc
        ans = math.floor(ans)
        decc = round(decc, 3)

        # 按 nameprice[name] 的值降序排序
        sorted_namelist = sorted(
            namelist, key=lambda name: nameprice[name], reverse=True
        )
        txt = ""

        # gang = ""

        # for name in sorted_namelist:
        #     gang = ""
        #     len_gang = (20 - len(name))*2
        #     for _ in range(len_gang):
        #         gang += "-"
        #     txt = txt+f"{name}:{gang}{nameprice[name]}\n"

        txt = "当前所标注高价值皮肤有：\n"

        for ind, name in enumerate(sorted_namelist):
            txt_line = f"{ind + 1}   {name}:    {nameprice[name]}\n"
            txt = txt + txt_line

        txt = (
            txt
            + f"所标注皮肤总价格为{price_ori}\n建议乘折扣系数{decc}\n得到基础价格{ans}\n"
        )
        return txt, ans

    def compute_only(total_count: int, price_ori: int):
        decc = 0.97**total_count
        if decc <= 0.3:
            decc = 0.3
        decc = math.log10(10 * decc)
        if decc <= 0.65:
            decc = 0.65
        ans = price_ori * decc
        ans = math.floor(ans)
        decc = round(decc, 3)

        txt = f"建议乘折扣系数{decc}\n得到基础价格{ans}\n"
        return txt, ans
