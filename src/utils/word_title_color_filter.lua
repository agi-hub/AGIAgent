-- Word文档标题颜色过滤器（版本2）
-- 通过修改段落属性而不是依赖样式

function Header(el)
  -- 只对docx格式进行处理
  if FORMAT ~= 'docx' then
    return el
  end
  
  io.stderr:write("Title color filter v2: Processing header level " .. el.level .. "\n")
  
  -- 创建新的内容，将标题文本包装在具有颜色属性的Span中
  local new_content = {}
  
  for i, inline in ipairs(el.content) do
    if inline.t == "Str" then
      -- 为字符串创建带颜色的Span
      local span_attr = pandoc.Attr("", {}, {style = "color: #000000"})
      table.insert(new_content, pandoc.Span({inline}, span_attr))
    elseif inline.t == "Space" then
      -- 空格也需要包装
      local span_attr = pandoc.Attr("", {}, {style = "color: #000000"})
      table.insert(new_content, pandoc.Span({inline}, span_attr))
    else
      -- 其他内容保持不变
      table.insert(new_content, inline)
    end
  end
  
  -- 如果没有处理任何内容，使用原内容
  if #new_content == 0 then
    new_content = el.content
  end
  
  -- 创建新的标题属性
  local new_attributes = {}
  if el.attr.attributes then
    for k, v in pairs(el.attr.attributes) do
      new_attributes[k] = v
    end
  end
  
  -- 添加颜色属性
  new_attributes.style = "color: #000000; color: black"
  
  local new_attr = pandoc.Attr(
    el.attr.identifier or "", 
    el.attr.classes or {}, 
    new_attributes
  )
  
  io.stderr:write("Title color filter v2: Applied black color to header\n")
  
  -- 返回修改后的标题元素
  return pandoc.Header(el.level, new_content, new_attr)
end

return {{Header = Header}}
