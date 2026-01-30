-- Word文档SVG转PNG过滤器
-- 用于在Word转换时将SVG图片路径替换为PNG路径，以提高兼容性
-- 只修改扩展名，保持路径不变

function Image(el)
  -- 只对docx格式进行处理
  if FORMAT ~= 'docx' then
    return el
  end
  
  -- 获取图片源路径
  local src = el.src
  
  -- 检查是否为SVG文件
  if src and src:match("%.svg$") then
    -- 将.svg扩展名替换为.png
    local png_src = src:gsub("%.svg$", ".png")
    
    -- 创建新的图片元素，保持其他属性不变
    return pandoc.Image(el.caption, png_src, el.title, el.attr)
  end
  
  -- 非SVG图片保持不变
  return el
end

return {{Image = Image}}
