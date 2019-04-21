const fs = require('fs')
const Octokit = require('@octokit/rest').plugin(require('@octokit/plugin-throttling'))
var path = require('path');

const octokit = new Octokit({
  auth: "token 322638e9a027580ff9743f07038acfc7e9169040",
  throttle: {
    onRateLimit: (retryAfter, options) => {
      octokit.log.warn(`Request quota exhausted for request ${options.method} ${options.url}`)

      if (options.request.retryCount === 0) { // only retries once
        console.log(`Retrying after ${retryAfter} seconds!`)
        return true
      }
    },
    onAbuseLimit: (retryAfter, options) => {
      // does not retry, only logs a warning
      octokit.log.warn(`Abuse detected for request ${options.method} ${options.url}`)
    }
  }
})





//  function add_usernames(rootuser) {
//   // Compare: https://developer.github.com/v3/repos/#list-organization-repositories

//   // octokit.users.listFollowingForUser({
//   //   username: rootuser
//   // }).then(({ data, status, headers }) => {

//   //   console.log("Hello")
//   //   if (usernames.length<128){
//   //     add_usernames(usernames[usernames.length-1])
//   //   }
//   //   console.log("More than 128 now")
//   // })


//   octokit.users.listFollowingForUser({
//     username: rootuser
//   }).then(({ data, status, headers }) => {
//     expanded.add(rootuser)

//     for (var da of data) {
//       name = da["login"]
//       usernames.push(name)
//     }

//     if (usernames.length<min_length){
//       let user_idx=usernames.length-1 
//       next_user=usernames[user_idx]
//       while (expanded.has(next_user)){
//         user_idx--
//         next_user=usernames[user_idx]
//       }
//       add_usernames(next_user)
//     }
//   }).catch((error)=>{
//     console.log(error)
//   })
// }

// initialize the usernames
// for (var i=0; i<16; i++){
//   var thisuser = usernames[usernames.length-1]
//   add_usernames(thisuser, usernames)
// }

// initialize the usernames

async function scrape() {
  var epochs = 10
  var usernames = read_stack()
  var expanded = new Set()
  var clone_lang = []
  const min_length = 16

  await initialize_usernames(usernames, min_length, expanded)

  for (let e = 0; e < epochs; e++) {
    // const result = await octokit.rateLimit.get({})
    // console.log(result)
    await scrape_one_epoch(usernames, expanded, clone_lang, min_length)

    //write the clone_lang and initiate new ones
    fname=await get_new_file_name()
    console.log(fname)
    await fs.writeFileSync(
      fname,
      JSON.stringify(clone_lang),
      function (err) {
        if (err) {
          console.error('Crap happens')
        }
      }
    )

    // dump stack
    await fs.writeFileSync(
      path.dirname(require.main.filename)+"\\stack",
      JSON.stringify(usernames),
      function (err) {
        if (err) {
          console.error('Crap happens')
        }
      }
    )
    clone_lang = []
  }
}

async function scrape_one_epoch(usernames, expanded, clone_lang, min_length, run_iter = 100) {
  // bfs, doesn't matter
  for (let i = 0; i < run_iter; i++) {
    console.log("initializing usernames")
    console.log("got ", usernames.length, "usernames")
    console.log("collect a user's repositories")
    await get_repos(usernames, clone_lang)
    test_and_add(usernames, expanded, min_length)
    console.log("current repos length: ", clone_lang.length)
  }
}

async function get_repos(usernames, clone_lang) {
  username = usernames.pop(0)
  let result = await octokit.repos.listForUser({ username: username })
  var repos = result.data
  for (repo of repos) {
    var url = repo["clone_url"]
    var lang = repo["language"]
    clone_lang.push([url, lang])
  }
}

async function initialize_usernames(usernames, min_length, expanded) {
  let i = 0
  while (usernames.length < min_length) {
    // wait for the promise to resolve before advancing the for loop
    // let results = await octokit.users.listFollowingForUser({
    //   username: next_user
    // })
    // let data=results.data
    // console.log("Something")
    // for (var da of data) {
    //   name = da["login"]
    //   usernames.push(name)
    // }

    // expanded.add(next_user)

    await add_user(usernames, expanded)

    // let user_idx=usernames.length-1 
    //   next_user=usernames[user_idx]
    //   while (expanded.has(next_user)){
    //     user_idx--
    //     next_user=usernames[user_idx]
    //   }
    i++
  }
}


async function add_user(usernames, expanded) {
  // find a user that is not expanded before
  user_idx = usernames.length - 1
  next_user = usernames[user_idx]
  while (expanded.has(next_user)) {
    user_idx--
    next_user = usernames[user_idx]
  }

  let results = await octokit.users.listFollowingForUser({
    username: next_user
  })
  let data = results.data
  for (var da of data) {
    name = da["login"]
    usernames.push(name)
  }

  expanded.add(next_user)
}

function test_and_add(usernames, expanded, min_length) {
  if (usernames.length < min_length) {
    add_user(usernames, expanded)
  }
}

// scrape()

// async function get_new_file_name(callback) {
//   var appDir = path.dirname(require.main.filename)+"\\data"
//   var highest_index=0

//   retval= fs.readdirSync(appDir, function (err, items) {
//     for (var i = 0; i < items.length; i++) {
//       let splitted=items[i].split('.')
//       let index=parseInt(splitted[0])
//       if (index>highest_index){
//         highest_index=index
//       }
//     }
//     var fname = appDir+"\\data\\"+highest_index+".json"
//     callback(null, fname)
//   });
// }

async function get_new_file_name_helper() {
  var appDir = path.dirname(require.main.filename)+"\\data\\"
  var highest_index=0

  var items= fs.readdirSync(appDir)
  
  for (var i = 0; i < items.length; i++) {
    let splitted=items[i].split('.')
    let index=parseInt(splitted[0])
    if (index>highest_index){
      highest_index=index
    }
  }
  highest_index++
  var fname = appDir+highest_index+".json"
  return fname
}


async function get_new_file_name(){
  fname=await get_new_file_name_helper()
  return fname
}


function read_stack(){
  const stack_location=path.dirname(require.main.filename)+"\\stack"
  var result= fs.readFileSync(stack_location)
  var array=JSON.parse(result)
  console.log("Check this")
  return array
}
scrape()

// read_stack()