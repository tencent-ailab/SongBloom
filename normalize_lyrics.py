import re

def clean_lyrics(text):
    structure_tags = ["intro", "outro", "verse", "chorus", "bridge", "inst", "silence"]
    tag_pattern = r"\[(" + "|".join(structure_tags) + r")\]"
    
    
    # 替换省略号、其他标点 → .
    text = re.sub(r"[!?;,:…！？，]", ".", text)
    # ① 暂存结构标识后面的逗号（防止被替换掉）
    text = re.sub(rf"({tag_pattern})\s*[,.]", r"\1 ", text, flags=re.IGNORECASE)
    
    # ② 按结构标识切分
    parts = re.split(r"(\[[^\]]+\])", text)
    cleaned_parts = []

    
    for part in parts:
        if not part.strip():
            continue
        
        # 如果是结构标识
        if re.fullmatch(tag_pattern, part.strip(), re.IGNORECASE):
            cleaned_parts.append(part.strip().lower() + " ")
        else:
            # 去除引号、书名号、括号
            part = re.sub(r"[\"“”‘’«»〈〉《》【】\(\)\{\}]", "", part)

            # 规范空格
            part = re.sub(r"\s+", " ", part.strip())
            cleaned_parts.append(part.strip())
    
    # ③ 构造输出（插入逗号逻辑）
    result_parts = []
    for i, seg in enumerate(cleaned_parts):
        result_parts.append(seg)
        
        if i < len(cleaned_parts) - 1:
            curr = cleaned_parts[i].strip()
            nxt = cleaned_parts[i + 1].strip()
            
            is_curr_tag = re.fullmatch(tag_pattern, curr.split()[0]) if curr else False
            is_next_tag = re.fullmatch(tag_pattern, nxt.split()[0]) if nxt else False

            if is_curr_tag and is_next_tag:
                # 连续相同结构不加逗号
                if curr.split()[0].lower() == nxt.split()[0].lower():
                    continue
                else:
                    result_parts.append(" , ")
            elif is_curr_tag or is_next_tag:
                if curr not in ['[verse]', '[chorus]', '[bridge]']:
                    result_parts.append(" , ")
    
    # ④ 清理格式
    result = "".join(result_parts)
    result = re.sub(r"\s+,", " ,", result)
    result = re.sub(r",\s+", ", ", result)
    result = re.sub(r"\.{2,}", ".", result)
    result = re.sub(r"\.\s*,", ",", result)
    result = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1.\2", result)
    
    return result.strip()
