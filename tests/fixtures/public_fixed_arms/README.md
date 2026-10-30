Exact public text bytes at the pinned commits of fanuc-corporation/fanuc_description
(fb40c98), ros-industrial/motoman (846fbbc), AgileRobots-APAC/diana_description (9278666),
Kinovarobotics/kinova-ros (924781d) and kuka-ros/kuka_robot_descriptions (5ddbc41), with
their license and notice files. inventory.json records each retained path, size and Git
blob identity; text rows also carry SHA-256 and are checked byte-for-byte. Mesh rows are
synthesized as development-only geometry by tests and were not downloaded for this
fixture, so meshes too large for a partial clone carry a null SHA-256. Keep publisher
whitespace and CRLF line endings unchanged. runtime_configuration_867c.py is the exact
867cee8 generator, used only to test retained 31-profile receipt reuse, and
predecessor_identities.json pins the 37 recipes, 112 catalog rows, scripted profile and
20 module hashes that existed before this batch. Nothing here is native, GPU or physical
validation.
