-- Execute the actual old/new Lua helpers against deterministic group fixtures.
-- Role predicates are pure card/target/substitute tests, as required by fusion.
local function helper(path)
  local file=assert(io.open(path,"rb")); local text=file:read("*a"); file:close()
  local first=assert(text:find("function Auxiliary.FCheckMixRep(sg",1,true))
  local last=assert(text:find("---Fusion monster, name + name",first,true))
  local env=setmetatable({Auxiliary={},Group={},Duel={},PLAYER_NONE=255,
    EFFECT_TUNE_MAGICIAN_F=1,EFFECT_MUST_BE_FMATERIAL=2,LOCATION_MZONE=4}, {__index=_G})
  local A=env.Auxiliary
  local methods={}
  local function ordered(g)
    local t={}; for c in pairs(g.members) do t[#t+1]=c end
    table.sort(t,function(a,b) return a.id<b.id end); return t
  end
  function env.Group.CreateGroup()
    return setmetatable({members={}}, {__index=methods})
  end
  function methods:AddCard(c) self.members[c]=true end
  function methods:RemoveCard(c) self.members[c]=nil end
  function methods:IsContains(c) return self.members[c]~=nil end
  function methods:Clone()
    local g=env.Group.CreateGroup()
    for c in pairs(self.members) do g:AddCard(c) end
    return g
  end
  function methods:Sub(g) for c in pairs(g.members) do self:RemoveCard(c) end end
  function methods:GetCount() local n=0; for _ in pairs(self.members) do n=n+1 end; return n end
  function methods:GetFirst() self.iter=ordered(self); self.at=1; return self.iter[1] end
  function methods:GetNext() self.at=self.at+1; return self.iter[self.at] end
  local function excluded(c,except)
    return except and (except.members and except:IsContains(c) or except==c)
  end
  function methods:IsExists(f,n,except,...)
    env.calls=env.calls+1; assert(env.calls<10000000,"fixture budget exceeded")
    local hits=0
    for _,c in ipairs(ordered(self)) do
      if not excluded(c,except) and f(c,...) then
        hits=hits+1; if hits>=n then return true end
      end
    end
    return false
  end
  function methods:FilterCount(f,except,...)
    local hits=0
    for _,c in ipairs(ordered(self)) do
      if not excluded(c,except) and f(c,...) then hits=hits+1 end
    end
    return hits
  end
  A.TuneMagicianCheckX=function() return false end
  A.MustMaterialCheck=function(g) return not env.must or g:IsContains(env.must) end
  env.Duel.GetLocationCountFromEx=function(tp,p,sg)
    return (not env.zone or sg:IsContains(env.zone)) and 1 or 0
  end
  assert(load(text:sub(first,last-1),"@"..path,"t",env))()
  return env
end

local old,new=helper(arg[1]),helper(arg[2])
local function role(i)
  return function(c,fc,sub)
    return c.kinds[i]==1 or (sub and c.kinds[i]==2)
  end
end
local function run(env,cards,selected,candidate,minc,maxc,roles,sub,zone,must)
  env.calls=0; env.zone=zone; env.must=must
  local mg,sg=env.Group.CreateGroup(),env.Group.CreateGroup()
  for _,c in ipairs(cards) do mg:AddCard(c) end
  for _,c in ipairs(selected) do sg:AddCard(c) end
  local before_first=sg:GetFirst()
  local before_at=sg.at
  local result=env.Auxiliary.FSelectMixRep(candidate,0,mg,sg,{},sub,0,
    role(1),minc,maxc,table.unpack(roles))
  assert(sg:GetCount()==#selected,"selected group not restored")
  assert(sg.at==before_at and sg.iter[1]==before_first,"selected iterator changed")
  for _,c in ipairs(selected) do assert(sg:IsContains(c),"selected identity changed") end
  return not not result,env.calls
end

math.randomseed(20261002)
for case=1,6000 do
  local cards,selected,roles={},{},{}
  for i=1,math.random(2,7) do
    local c={id=i,kinds={math.random(0,2),math.random(0,2),math.random(0,2)}}
    function c:IsControler(tp) return true end
    function c:IsLocation(loc) return self.id%2==0 end
    cards[#cards+1]=c
    if i>1 and math.random(0,1)==1 then selected[#selected+1]=c end
  end
  for i=1,math.random(0,2) do roles[#roles+1]=role(i+1) end
  local minimum=math.random(0,3); local maximum=minimum+math.random(0,3)
  local sub=math.random(0,1)==1
  local zone=math.random(0,1)==1 and cards[math.random(#cards)] or nil
  local must=math.random(0,1)==1 and cards[math.random(#cards)] or nil
  local a=run(old,cards,selected,cards[1],minimum,maximum,roles,sub,zone,must)
  local b=run(new,cards,selected,cards[1],minimum,maximum,roles,sub,zone,must)
  assert(a==b,"old/new legal-material disagreement at case "..case)
end

-- Already-selected repeated-role cards cannot meet a missing mandatory card.
-- The old helper visits permutations; the new helper still checks all roles.
local cards,selected={},{}
for i=1,9 do
  local c={id=i,kinds={1,0,0}}
  cards[i]=c; if i>1 then selected[#selected+1]=c end
end
local missing={id=100,kinds={0,0,0}}
local a,oldcalls=run(old,cards,selected,cards[1],1,127,{},false,nil,missing)
local b,newcalls=run(new,cards,selected,cards[1],1,127,{},false,nil,missing)
assert(not a and a==b)
assert(oldcalls>100000 and newcalls<100,"factorial path was not removed")
print(string.format('{"equivalence_cases":6000,"mismatches":0,"old_calls":%d,"new_calls":%d}',oldcalls,newcalls))
