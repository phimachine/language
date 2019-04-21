
var path = require('path')
const fs = require('fs')

function find_users(found_users){
  // find the users from the items
  var itemDir = path.dirname(require.main.filename)+"\\items\\"
  var highest_index=0

  var jsonfiles= fs.readdirSync(itemDir)
  
  for (var i = 0; i < jsonfiles.length; i++) {
    let jsonfile=jsonfiles[i]
    let fpath=itemDir+jsonfile
    
    try{
      var result= fs.readFileSync(fpath)
    } catch (error){
      console.error(error)
    }
    var items=JSON.parse(result)
    
    for (item of items){
      let username=item[0]
      let splitted=username.split("/")
      username=splitted[splitted.length-2]
      found_users.add(username)
    }
  }
  return found_users
}

module.exports={
  find_users:find_users
}