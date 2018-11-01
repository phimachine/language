This demo is to demonstrate that WebAssembly is a possible option for our compiler solution.


Previously in our team discussion, the point has been raised that WebAssembly consumes too much resources, in client side, server side or disk usage. Turns out none of that is true. This is a misunderstanding due to my teammate misinterpreting some Reddit comments about WebAssembly.

This is a hello world program that I have compiled, according to the official guide. https://webassembly.org/getting-started/developers-guide/

It's easy to see that the whole compiled WA program is under 300kb in total, including HTML.

Chrome benchmark shows that the website consumes the normal amount of resources that it should consume.

Because I cannot serve this website due to CORS, I used emrun to run a basic server to serve the webassembly website. The server costs around 20MB to run. We will have a Flask server, so serving this website will eventually have a much smaller overhead.


### To run
To run the program, open hello.html.
If CORS blocks you, circumvent it somehow. For example, $ emrun --no_browser --port 8080 .

To compile the c program, see compile.bash

For performance measurement, see the photos included in the folder.

If you have any questions, read the guide above.
