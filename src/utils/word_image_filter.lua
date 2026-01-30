-- Word文档图片大小限制过滤器（pandoc 2.5兼容版本）
-- 基于pandoc 2.5的具体实现方式

function Image(el)
  -- 只对docx格式进行处理
  if FORMAT ~= 'docx' then
    return el
  end
  
  -- 设置最大尺寸限制（使用英寸单位，pandoc 2.5对此支持更好）
  local max_height_in = "9.4in"  -- 约24cm
  local max_width_in = "6.3in"   -- 约16cm
  
  -- 检查是否需要限制
  local orig_width = el.attr.attributes.width
  local orig_height = el.attr.attributes.height
  
  local need_limit = false
  
  if not orig_height and not orig_width then
    need_limit = true
  elseif orig_height then
    -- 检查现有高度
    local height_num = orig_height:match("([%d%.]+)")
    if height_num then
      height_num = tonumber(height_num)
      local height_unit = orig_height:match("[%a%%]+") or "px"
      
      -- 转换为英寸进行比较
      local height_in_inches = height_num
      if height_unit == "cm" then
        height_in_inches = height_num / 2.54
      elseif height_unit == "px" then
        height_in_inches = height_num / 96
      elseif height_unit == "pt" then
        height_in_inches = height_num / 72
      end
      
      if height_in_inches > 9.4 then
        need_limit = true
      end
    end
  end
  
  if need_limit then
    -- 创建新的属性表，确保兼容性
    local new_attributes = {}
    
    -- 复制原有属性，排除尺寸相关的
    for k, v in pairs(el.attr.attributes) do
      if k ~= "width" and k ~= "height" and k ~= "style" then
        new_attributes[k] = v
      end
    end
    
    -- 设置新的尺寸（使用英寸单位）
    new_attributes.height = max_height_in
    -- 不设置宽度，让Word自动保持长宽比
    
    -- 对于pandoc 2.5，还可以尝试设置额外的属性
    local new_classes = {}
    for _, class in ipairs(el.attr.classes) do
      table.insert(new_classes, class)
    end
    table.insert(new_classes, "size-limited")
    
    -- 创建新的Attr对象
    local new_attr = pandoc.Attr(el.attr.identifier or "", new_classes, new_attributes)
    
    -- 返回新的图片元素
    return pandoc.Image(el.caption, el.src, el.title, new_attr)
  end
  
  return el
end

return {{Image = Image}}
