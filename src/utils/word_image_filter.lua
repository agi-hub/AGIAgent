-- Word文档图片大小限制过滤器
-- 用于限制Word文档中图片的最大高度，避免图片过高，同时保持长宽比

function Image(el)
  -- 只对docx格式进行处理
  if FORMAT ~= 'docx' then
    return el
  end
  
  -- 获取图片的原始属性
  local attr = el.attr
  local width = attr.attributes.width
  local height = attr.attributes.height
  
  -- 设置最大尺寸限制
  local max_height = "24cm"  -- 约等于A4页面全页高度（减去边距）
  local max_width = "16cm"   -- 约等于A4页面宽度减去边距
  
  -- 检查是否需要限制高度
  local need_height_limit = false
  
  if height and not height:match("%%") then
    -- 如果高度超过最大值，需要限制
    local height_num = height:match("([%d%.]+)")
    local height_unit = height:match("[%a%%]+")
    
    if height_num and height_unit then
      height_num = tonumber(height_num)
      -- 检查是否超过限制
      if (height_unit == "cm" and height_num > 24) or
         (height_unit == "px" and height_num > 900) or
         (height_unit == "in" and height_num > 9.4) then
        need_height_limit = true
      end
    end
  end
  
  -- 应用尺寸限制，优先保持长宽比
  if not height and not width then
    -- 没有设置任何尺寸，使用最大高度约束，保持长宽比
    attr.attributes.style = "max-height: " .. max_height .. "; max-width: " .. max_width .. "; height: auto; width: auto;"
  elseif need_height_limit then
    -- 需要限制高度，移除原有尺寸设置，使用CSS样式保持长宽比
    attr.attributes.height = nil
    attr.attributes.width = nil
    attr.attributes.style = "max-height: " .. max_height .. "; max-width: " .. max_width .. "; height: auto; width: auto;"
  elseif not height then
    -- 只设置了宽度，添加最大高度约束
    attr.attributes.style = (attr.attributes.style or "") .. "; max-height: " .. max_height .. ";"
  elseif not width then
    -- 只设置了高度，添加最大宽度约束
    attr.attributes.style = (attr.attributes.style or "") .. "; max-width: " .. max_width .. ";"
  end
  
  -- 返回修改后的图片元素
  return pandoc.Image(el.caption, el.src, el.title, attr)
end

return {{Image = Image}}